#!/usr/bin/env python3
"""
Simple 3D Phone Visualizer
Shows a phone upright and facing the viewer, rotating with real-time sensor data.
"""

import socket
import threading
import json
import time
import sys
import math

try:
    import pygame
    from pygame.locals import *
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Error: pygame not installed. Run: pip3 install pygame")
    sys.exit(1)

try:
    from OpenGL.GL import *
    try:
        from OpenGL.GLU import *
    except:
        pass
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    print("Error: PyOpenGL not installed. Run: pip3 install PyOpenGL")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Error: numpy not installed. Run: pip3 install numpy")
    sys.exit(1)


# Global sensor data
sensor_data = {
    'roll': 0.0,
    'pitch': 0.0,
    'yaw': 0.0,
    'quat': [1.0, 0.0, 0.0, 0.0], # w, x, y, z
    'connected': False,
    'last_update': time.time()
}
data_lock = threading.Lock()


def quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles (in degrees) to quaternion."""
    # Convert to radians
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)
    
    # Calculate quaternion components
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return [qw, qx, qy, qz]


def quaternion_to_matrix(q):
    """Convert quaternion to 4x4 rotation matrix."""
    w, x, y, z = q
    
    matrix = [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), 0],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), 0],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y), 0],
        [0, 0, 0, 1]
    ]
    return matrix


def apply_rotation_from_angles(roll_deg, pitch_deg, yaw_deg):
    """Apply rotation using direct Euler angles (Roll-Pitch-Yaw) without quaternion."""
    # Convert to radians
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
    # Apply rotations in order: first yaw (Z), then pitch (Y), then roll (X)
    # This matches the standard phone orientation convention
    glRotatef(yaw_deg, 0, 0, 1)      # Rotate around Z axis (yaw)
    glRotatef(pitch_deg, 0, 1, 0)    # Rotate around Y axis (pitch)
    glRotatef(roll_deg, 1, 0, 0)     # Rotate around X axis (roll)


def apply_rotation_from_quaternion(quat):
    """Apply rotation from quaternion [w, x, y, z]."""
    w, x, y, z = quat
    
    # Calculate rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Column-major matrix for OpenGL
    # [ R00, R10, R20, 0 ]
    # [ R01, R11, R21, 0 ]
    # [ R02, R12, R22, 0 ]
    # [ 0,   0,   0,   1 ]
    matrix = [
        1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy),       0.0,
        2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),       0.0,
        2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy), 0.0,
        0.0,                   0.0,                   0.0,                   1.0
    ]
    
    glMultMatrixf(matrix)


def draw_phone():
    """Draw a simple phone model (rectangular box) with labeled faces."""
    # Phone dimensions (width, height, depth in normalized units)
    w = 0.4  # width
    h = 0.8  # height
    d = 0.08  # depth (thin)
    
    # Define phone faces with colors and labels
    faces = [
        # Front face (screen) - bright blue
        {
            'vertices': [[-w/2, -h/2, d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [-w/2, h/2, d/2]],
            'color': (0.1, 0.3, 0.8),  # Blue for screen
            'label': 'SCREEN'
        },
        # Back face - dark red
        {
            'vertices': [[-w/2, -h/2, -d/2], [-w/2, h/2, -d/2], [w/2, h/2, -d/2], [w/2, -h/2, -d/2]],
            'color': (0.6, 0.1, 0.1),  # Red for back
            'label': 'BACK'
        },
        # Left face - green
        {
            'vertices': [[-w/2, -h/2, d/2], [-w/2, -h/2, -d/2], [-w/2, h/2, -d/2], [-w/2, h/2, d/2]],
            'color': (0.1, 0.6, 0.1),  # Green for left
            'label': 'LEFT'
        },
        # Right face - yellow
        {
            'vertices': [[w/2, -h/2, -d/2], [w/2, -h/2, d/2], [w/2, h/2, d/2], [w/2, h/2, -d/2]],
            'color': (0.8, 0.8, 0.1),  # Yellow for right
            'label': 'RIGHT'
        },
        # Top face - cyan
        {
            'vertices': [[-w/2, h/2, d/2], [w/2, h/2, d/2], [w/2, h/2, -d/2], [-w/2, h/2, -d/2]],
            'color': (0.1, 0.8, 0.8),  # Cyan for top
            'label': 'TOP'
        },
        # Bottom face - magenta
        {
            'vertices': [[-w/2, -h/2, -d/2], [w/2, -h/2, -d/2], [w/2, -h/2, d/2], [-w/2, -h/2, d/2]],
            'color': (0.8, 0.1, 0.8),  # Magenta for bottom
            'label': 'BOTTOM'
        },
    ]
    
    # Draw each face
    glBegin(GL_QUADS)
    for face in faces:
        glColor3f(*face['color'])
        for vertex in face['vertices']:
            glVertex3f(*vertex)
    glEnd()
    
    # Draw edges in black
    glColor3f(0, 0, 0)
    glLineWidth(2.0)
    for face in faces:
        glBegin(GL_LINE_LOOP)
        for vertex in face['vertices']:
            glVertex3f(*vertex)
        glEnd()
    glLineWidth(1.0)


def draw_reference_axes():
    """Draw RGB reference axes at the origin."""
    axis_length = 1.5
    
    glLineWidth(2.0)
    glBegin(GL_LINES)
    
    # X axis - Red
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(axis_length, 0, 0)
    
    # Y axis - Green
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, axis_length, 0)
    
    # Z axis - Blue
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, axis_length)
    
    glEnd()
    glLineWidth(1.0)


def udp_receiver():
    """Receive sensor data via UDP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 3401))
        sock.settimeout(1.0)
        
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                
                try:
                    decoded_data = data.decode('utf-8')
                    # print(f"Raw data: {decoded_data}") # Uncomment to debug raw JSON
                    
                    json_data = json.loads(decoded_data)
                    ori = json_data.get('ori', [0, 0, 0])
                    quat = json_data.get('quat', [1, 0, 0, 0])
                    
                    with data_lock:
                        sensor_data['roll'] = ori[0]
                        sensor_data['pitch'] = ori[1]
                        sensor_data['yaw'] = ori[2]
                        sensor_data['quat'] = quat
                        sensor_data['connected'] = True
                        sensor_data['last_update'] = time.time()
                
                except json.JSONDecodeError as e:
                    print(f"JSON Error: {e} in data: {data}")
                    pass
            
            except socket.timeout:
                # Check connection timeout
                with data_lock:
                    if (time.time() - sensor_data['last_update']) > 2.0:
                        sensor_data['connected'] = False
    
    except Exception as e:
        print(f"UDP Receiver error: {e}")


def main():
    """Main visualizer loop."""
    print("\n" + "="*70)
    print("📱 PHONE 3D VISUALIZER - Real-time Sensor Stream")
    print("="*70)
    print("\nInitializing display...")
    
    # Initialize pygame
    pygame.init()
    display = (1200, 800)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Phone 3D Visualizer - Rotating with Sensor Data")
    
    # Initialize OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Setup lighting
    light_pos = [2, 2, 2, 0]
    glLight(GL_LIGHT0, GL_POSITION, light_pos)
    glLight(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
    glLight(GL_LIGHT0, GL_DIFFUSE, [1, 1, 1, 1])
    glLight(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
    
    # Setup perspective
    glMatrixMode(GL_PROJECTION)
    aspect = display[0] / display[1]
    fov = 45.0
    near = 0.1
    far = 50.0
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    glOrtho(-1 * aspect, 1 * aspect, -1, 1, near, far)
    glMatrixMode(GL_MODELVIEW)
    
    # White background
    glClearColor(1, 1, 1, 1)
    
    # Start UDP receiver thread
    receiver_thread = threading.Thread(target=udp_receiver, daemon=True)
    receiver_thread.start()
    
    print("✅ Display initialized")
    print("📊 Listening on UDP 127.0.0.1:3401")
    print("🎯 Close window or press ESC to exit\n")
    
    clock = pygame.time.Clock()
    frame_count = 0
    start_time = time.time()
    
    running = True
    while running:
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Get current sensor data
        with data_lock:
            roll = sensor_data['roll']
            pitch = sensor_data['pitch']
            yaw = sensor_data['yaw']
            quat = sensor_data['quat']
            connected = sensor_data['connected']
        
        # Setup camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, 0, -2.5)  # Move camera back to see phone
        
        # Draw reference axes
        draw_reference_axes()
        
        # Apply rotation from sensor data
        glPushMatrix()
        
        # Use quaternion directly for best accuracy and to avoid gimbal lock
        # Swap Y and Z to match the visualizer's coordinate system
        # Phone Z (Yaw) needs to map to Visualizer Y (Vertical spin)
        # Phone Y (Roll) needs to map to Visualizer Z (Longitudinal roll)
        # Negate Y (Roll) to fix inverted rotation
        w, x, y, z = quat
        quat_mapped = [w, x, z, -y]
        apply_rotation_from_quaternion(quat_mapped)
        
        # Rotate phone model to lie flat by default (instead of standing up)
        # The model is defined with height along Y. We want it along Z (or flat on XZ plane).
        # -90 degrees around X axis makes the top of the phone point away from the camera (into the screen)
        glRotatef(-90, 1, 0, 0)
        
        # Draw phone
        draw_phone()
        
        glPopMatrix()
        
        # Display FPS and status
        elapsed = time.time() - start_time
        if frame_count % 60 == 0 and elapsed > 0:
            fps = frame_count / elapsed
            status = "🔴 CONNECTED" if connected else "⚫ WAITING"
            print(f"{status} | Roll: {roll:7.2f}° Pitch: {pitch:7.2f}° Yaw: {yaw:7.2f}° | FPS: {fps:.1f}")
        
        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS target
    
    # Cleanup
    pygame.quit()
    print("\n✅ Visualizer closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ Visualizer closed by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
