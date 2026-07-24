# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebXR server node for OpenArm teleoperation.

This dora-rs node serves the WebXR front-end over HTTPS and accepts a
WebSocket connection from a VR device such as Meta Quest 3 or PICO 4.
For each frame received from the device, it converts the controller
pose from WebXR coordinates into the OpenArm workspace, smooths it with
a One Euro filter, and publishes the pose, trigger, joystick and button
state as dora-rs outputs.

The Web server and the dora-rs event loop run concurrently in a single
asyncio event loop; the server shuts down when the dora-rs node
receives a ``STOP`` event.
"""

import argparse
import asyncio
import dora
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import json
import numpy as np
import os
import pathlib
import pyarrow as pa
from scipy.spatial.transform import Rotation
import time
import uvicorn

from .smoothing import OneEuroPoseSmoother

args = None
node = None
server = None


# Relative pose to robot workspace mapping.
# We may need to adjust this.
_ROBOT_ROTATION_MATRIX: np.ndarray = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
_ROBOT_ROTATION = Rotation.from_matrix(_ROBOT_ROTATION_MATRIX)

# Aligns the controller grip frame with the OpenArm end effector
# frame. Inherited from the R_FIX of the Meta Quest UDP receiver that
# this node replaces.
_ROTATION_FIX = Rotation.from_euler("z", 90, degrees=True)

# Moves the viewer relative pose to the OpenArm position. The default
# is for the OpenArm Cell environment and mirrors FRAME_OFFSET_NECK of
# the Meta Quest UDP receiver.
_FRAME_OFFSET_DEFAULT = "0.1,0,1.2"


app = FastAPI()


def _map_trigger_to_gripper(trigger: float, side: str) -> float:
    """Trigger 0.0~1.0 -> gripper angle."""
    if side == "right":
        return (-1.57 / 2.0) * (1.0 - trigger)  # 0->-1.57, 1->0
    else:
        return (1.57 / 2.0) * (1.0 - trigger)  # 0-> 1.57, 1->0


def _rectify_pose(pose, reference, frame_offset):
    """Convert a WebXR controller pose to an OpenArm workspace pose.

    ``pose`` and ``reference`` (the viewer pose) are WebXR style poses
    in the same world-fixed reference space:
      * right-handed, Y-up
      * {
          x: X, (meter)
          y: Y, (meter)
          z: Z, (meter)
          qx: QX, (quaternion)
          qy: QY, (quaternion)
          qz: QZ, (quaternion)
          qw: QW, (quaternion)
        }

    The displacement from the viewer and the controller orientation
    are rotated into the viewer's own frame. This mirrors the NECK
    mode of the Meta Quest UDP receiver that this node replaces and is
    meant for a headset that rests on the operator's neck so that the
    frame follows the torso.

    Our style:
      * right-handed
      * [x, y, z, qw, qx, qy, qz]
    """
    position = np.array(
        [
            pose["x"] - reference["x"],
            pose["y"] - reference["y"],
            pose["z"] - reference["z"],
        ],
        dtype=np.float32,
    )
    rotation = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
    reference_rotation_inv = Rotation.from_quat(
        [reference["qx"], reference["qy"], reference["qz"], reference["qw"]]
    ).inv()
    position = reference_rotation_inv.apply(position)
    rotation = reference_rotation_inv * rotation
    position = _ROBOT_ROTATION.apply(position) + frame_offset
    rotation = _ROBOT_ROTATION * rotation * _ROTATION_FIX
    quaternion = rotation.as_quat()

    return np.array(
        [
            position[0],  # x
            position[1],  # y
            position[2],  # z
            quaternion[3],  # qw
            quaternion[0],  # qx
            quaternion[1],  # qy
            quaternion[2],  # qz
        ],
        dtype=np.float32,
    )


_POSE_STRUCT_TYPE = pa.struct({"pose": pa.list_(pa.float32())})


def _build_pose_output(pose: np.ndarray) -> pa.Array:
    """Wrap a pose array as a length-1 StructArray: [{"pose": [...]}]."""
    return pa.array([{"pose": pose}], type=_POSE_STRUCT_TYPE)


@app.websocket("/websocket")
async def _websocket_endpoint(websocket: WebSocket):
    smoothers = {
        "right": OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5),
        "left": OneEuroPoseSmoother(min_cutoff=2.0, beta=0.04, d_cutoff=1.5),
    }

    await websocket.accept()
    try:
        while not server.should_exit:
            data = await websocket.receive_text()
            response = json.loads(data)
            type = response["type"]
            metadata = {"timestamp": time.time_ns()}
            if type == "session-start":
                node.send_output("status", pa.array(["ready"]), metadata)
            elif type == "frame":
                smoother_time = time.perf_counter()
                node.send_output(
                    "vr_receive_times",
                    pa.array([metadata["timestamp"]], type=pa.int64()),
                    metadata,
                )
                for button in ["a", "b", "x", "y"]:
                    name = f"button_{button}"
                    if name in response:
                        node.send_output(
                            name,
                            pa.array([bool(response[name])], type=pa.bool_()),
                            metadata,
                        )
                reference = response.get("pose_reference")
                for side in ["right", "left"]:
                    pose = f"pose_{side}"
                    trigger = f"trigger_{side}"
                    if pose in response and trigger in response and reference:
                        smoother = smoothers[side]
                        adjusted_pose = smoother.smooth(
                            smoother_time,
                            _rectify_pose(response[pose], reference, args.frame_offset),
                        )
                        gripper_angle = _map_trigger_to_gripper(response[trigger], side)
                        gripper_array = np.array([gripper_angle], dtype=np.float32)
                        pose_with_gripper = np.concatenate(
                            [adjusted_pose, gripper_array]
                        )
                        node.send_output(
                            pose,
                            _build_pose_output(pose_with_gripper),
                            metadata,
                        )
                    if trigger in response:
                        node.send_output(
                            trigger,
                            pa.array([response[trigger]], type=pa.float32()),
                            metadata,
                        )
                    joystick = f"joystick_{side}"
                    if joystick in response:
                        axes = response[joystick]
                        x = axes[1] - axes[3]
                        y = axes[2] - axes[0]
                        node.send_output(
                            f"joystick_x_{side}",
                            pa.array([x], type=pa.float32()),
                            metadata,
                        )
                        node.send_output(
                            f"joystick_y_{side}",
                            pa.array([y], type=pa.float32()),
                            metadata,
                        )
        await websocket.close()
    except WebSocketDisconnect:
        pass


base_dir = os.path.dirname(__file__)
app.mount("/", StaticFiles(directory=f"{base_dir}/static", html=True), name="static")


async def _main_uvicorn():
    await server.serve()


async def _main_dora():
    while not server.should_exit:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            break
    server.should_exit = True


async def _main_async():
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.tls_key_file,
        ssl_certfile=args.tls_certificate_file,
        log_level="info",
    )
    global server
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn())
    task_dora = asyncio.create_task(_main_dora())

    await task_uvicorn
    await task_dora


def _parse_frame_offset(text):
    """Parse an ``x,y,z`` frame offset in meters as a NumPy array."""
    values = text.split(",")
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            f"frame offset must be 'x,y,z' in meters: {text!r}"
        )
    return np.array([float(value) for value in values], dtype=np.float32)


def main():
    """Run WebXR server."""
    parser = argparse.ArgumentParser(description="WebXR server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8443")),
        help="Server port (default: 8443)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)",
    )
    tls_certificate_file_default = os.getenv("TLS_CERTIFICATE_FILE")
    parser.add_argument(
        "--tls-certificate-file",
        type=pathlib.Path,
        default=tls_certificate_file_default,
        required=tls_certificate_file_default is None,
        help="TLS certificate file",
    )
    tls_key_file_default = os.getenv("TLS_KEY_FILE")
    parser.add_argument(
        "--tls-key-file",
        type=pathlib.Path,
        default=tls_key_file_default,
        required=tls_key_file_default is None,
        help="TLS key file for the certificate file",
    )
    parser.add_argument(
        "--frame-offset",
        type=_parse_frame_offset,
        default=os.getenv("FRAME_OFFSET", _FRAME_OFFSET_DEFAULT),
        help="Offset 'x,y,z' in meters that moves the viewer relative "
        f"pose to the OpenArm position (default: {_FRAME_OFFSET_DEFAULT})",
    )

    global args
    args = parser.parse_args()

    global node
    node = dora.Node()

    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
