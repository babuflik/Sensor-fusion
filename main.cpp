#include <iostream>
#include <thread>
#include <mutex>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdio>
#include <string>
#include <cmath>

#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #define closesocket close
#endif

// Try to include SensorFusion header if available
#ifdef __has_include
    #if __has_include(<SensorFusion/SensorFusion.h>)
        #include <SensorFusion/SensorFusion.h>
    #elif __has_include("SensorFusion.h")
        #include "SensorFusion.h"
    #endif
#endif

// Sensor data structure with additional fields for sensor fusion
struct SensorData {
    float accel[3];      // Accelerometer (x, y, z) m/s²
    float gyro[3];       // Gyroscope (x, y, z) rad/s
    float orientation[3]; // Orientation (roll, pitch, yaw) degrees
    float magfield[3];   // Magnetic field (x, y, z) μT
    float fusedQuaternion[4]; // Fused quaternion (qx, qy, qz, qw)
    std::mutex dataMutex;
};

SensorData sensorData = {};
bool running = true;
bool dataReceived = false;
std::mutex dataReceivedMutex;

// UDP socket for GUI visualization
static int guiUdpSocket = -1;
static struct sockaddr_in guiAddr;

// Complementary filter state
struct ComplementaryFilterState {
    float qx = 0, qy = 0, qz = 0, qw = 1;  // Current quaternion (identity)
    float lastUpdateTime = 0;
    float accelAlpha = 0.02f;  // Accelerometer weight (low-pass filter)
    float magAlpha = 0.01f;    // Magnetometer weight
} fusionState;

// Convert quaternion (qx, qy, qz, qw) to Euler angles (roll, pitch, yaw) in degrees
// Using ZYX convention (Yaw-Pitch-Roll)
void quaternionToEuler(float qx, float qy, float qz, float qw, float& roll, float& pitch, float& yaw) {
    const float PI = 3.14159265359f;
    
    // Normalize quaternion
    float norm = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    if (norm == 0) norm = 1;
    qx /= norm;
    qy /= norm;
    qz /= norm;
    qw /= norm;
    
    // Roll (x-axis rotation) - atan2(2*(w*x + y*z), 1 - 2*(x^2 + y^2))
    float sinr_cosp = 2 * (qw * qx + qy * qz);
    float cosr_cosp = 1 - 2 * (qx * qx + qy * qy);
    roll = atan2(sinr_cosp, cosr_cosp);
    
    // Pitch (y-axis rotation) - asin(2*(w*y - z*x))
    float sinp = 2 * (qw * qy - qz * qx);
    if (fabs(sinp) >= 1) {
        pitch = copysign(PI / 2, sinp);
    } else {
        pitch = asin(sinp);
    }
    
    // Yaw (z-axis rotation) - atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    float siny_cosp = 2 * (qw * qz + qx * qy);
    float cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    yaw = atan2(siny_cosp, cosy_cosp);
    
    // Convert from radians to degrees
    roll = roll * 180.0f / PI;
    pitch = pitch * 180.0f / PI;
    yaw = yaw * 180.0f / PI;
}

// Normalize vector
void normalizeVector(float& x, float& y, float& z) {
    float norm = sqrt(x*x + y*y + z*z);
    if (norm > 0.0001f) {
        x /= norm;
        y /= norm;
        z /= norm;
    }
}

// Get gravity vector from accelerometer
void getGravityFromAccel(float ax, float ay, float az, float& gx, float& gy, float& gz) {
    gx = ax;
    gy = ay;
    gz = az;
    normalizeVector(gx, gy, gz);
}

// Get down vector (opposite of gravity)
void getDownVector(float ax, float ay, float az, float& dx, float& dy, float& dz) {
    getGravityFromAccel(ax, ay, az, dx, dy, dz);
    // Down is opposite of up (gravity)
    dx = -dx;
    dy = -dy;
    dz = -dz;
}

// Quaternion multiplication
void quatMultiply(float q1x, float q1y, float q1z, float q1w,
                  float q2x, float q2y, float q2z, float q2w,
                  float& qx, float& qy, float& qz, float& qw) {
    qw = q1w*q2w - q1x*q2x - q1y*q2y - q1z*q2z;
    qx = q1w*q2x + q1x*q2w + q1y*q2z - q1z*q2y;
    qy = q1w*q2y - q1x*q2z + q1y*q2w + q1z*q2x;
    qz = q1w*q2z + q1x*q2y - q1y*q2x + q1z*q2w;
}

// Normalize quaternion
void normalizeQuaternion(float& qx, float& qy, float& qz, float& qw) {
    float norm = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
    if (norm > 0.0001f) {
        qx /= norm;
        qy /= norm;
        qz /= norm;
        qw /= norm;
    }
}

// Sensor fusion: update quaternion using gyroscope, accelerometer, and magnetometer
void updateSensorFusion(float ax, float ay, float az,
                        float gx, float gy, float gz,
                        float mx, float my, float mz,
                        float dt) {
    if (dt <= 0 || dt > 0.1f) return;  // Ignore invalid time deltas
    
    // 1. Integrate gyroscope (angular velocity) to get rotation
    float halfGx = gx * 0.5f * dt;
    float halfGy = gy * 0.5f * dt;
    float halfGz = gz * 0.5f * dt;
    
    // Create quaternion from angular velocity
    float dqx = halfGx;
    float dqy = halfGy;
    float dqz = halfGz;
    float dqw = 1.0f - (halfGx*halfGx + halfGy*halfGy + halfGz*halfGz) * 0.5f;
    normalizeQuaternion(dqx, dqy, dqz, dqw);
    
    // Apply gyro rotation
    quatMultiply(fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw,
                 dqx, dqy, dqz, dqw,
                 fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw);
    normalizeQuaternion(fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw);
    
    // 2. Correct using accelerometer (gravity reference)
    float gax = ax, gay = ay, gaz = az;
    normalizeVector(gax, gay, gaz);
    
    // Estimate gravity from current quaternion orientation
    float estGx = 2 * (fusionState.qx * fusionState.qz - fusionState.qw * fusionState.qy);
    float estGy = 2 * (fusionState.qy * fusionState.qz + fusionState.qw * fusionState.qx);
    float estGz = fusionState.qw*fusionState.qw - fusionState.qx*fusionState.qx - 
                  fusionState.qy*fusionState.qy + fusionState.qz*fusionState.qz;
    normalizeVector(estGx, estGy, estGz);
    
    // Cross product for correction
    float corrX = gay * estGz - gaz * estGy;
    float corrY = gaz * estGx - gax * estGz;
    float corrZ = gax * estGy - gay * estGx;
    
    // Apply correction with low-pass filter
    float corrScale = fusionState.accelAlpha;
    float corrQx = corrScale * corrX * 0.5f;
    float corrQy = corrScale * corrY * 0.5f;
    float corrQz = corrScale * corrZ * 0.5f;
    float corrQw = 1.0f;
    normalizeQuaternion(corrQx, corrQy, corrQz, corrQw);
    
    quatMultiply(fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw,
                 corrQx, corrQy, corrQz, corrQw,
                 fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw);
    normalizeQuaternion(fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw);
}
void initGuiBroadcast() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "Failed to initialize Winsock for GUI broadcast" << std::endl;
        return;
    }
#endif
    
    guiUdpSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (guiUdpSocket < 0) {
        std::cerr << "Failed to create UDP socket for GUI" << std::endl;
        return;
    }
    
    std::memset(&guiAddr, 0, sizeof(guiAddr));
    guiAddr.sin_family = AF_INET;
    guiAddr.sin_port = htons(3401);  // GUI visualizer port
    guiAddr.sin_addr.s_addr = inet_addr("127.0.0.1");  // Localhost only
}

// Send sensor data to GUI visualizer
void sendToGui() {
    if (guiUdpSocket < 0) return;
    
    try {
        std::lock_guard<std::mutex> lock(sensorData.dataMutex);
        
        // Create JSON: {"quat": [w,x,y,z], "ori": [roll, pitch, yaw], "acc": [x, y, z], "gyro": [x, y, z], "mag": [x, y, z]}
        std::ostringstream json;
        json << std::fixed << std::setprecision(4);
        json << "{\"quat\":[" << sensorData.fusedQuaternion[3] << "," 
                              << sensorData.fusedQuaternion[0] << "," 
                              << sensorData.fusedQuaternion[1] << "," 
                              << sensorData.fusedQuaternion[2] << "],"
             << "\"ori\":[" << sensorData.orientation[0] << "," 
                             << sensorData.orientation[1] << "," 
                             << sensorData.orientation[2] << "],"
             << "\"acc\":[" << sensorData.accel[0] << "," 
                             << sensorData.accel[1] << "," 
                             << sensorData.accel[2] << "],"
             << "\"gyro\":[" << sensorData.gyro[0] << "," 
                              << sensorData.gyro[1] << "," 
                              << sensorData.gyro[2] << "],"
             << "\"mag\":[" << sensorData.magfield[0] << "," 
                             << sensorData.magfield[1] << "," 
                             << sensorData.magfield[2] << "]}";
        
        std::string jsonStr = json.str();
        
        // Debug print JSON occasionally
        /*
        static int json_debug_count = 0;
        if (json_debug_count++ % 100 == 0) {
            std::cout << "[DEBUG] Sending JSON: " << jsonStr << std::endl;
        }
        */
        
        int sent = sendto(guiUdpSocket, jsonStr.c_str(), jsonStr.length(), 0,
                         (struct sockaddr*)&guiAddr, sizeof(guiAddr));
        
        if (sent < 0) {
            static int errorCount = 0;
            if (errorCount++ % 100 == 0) {
                std::cerr << "[GUI] Failed to send data to visualizer" << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[GUI] Error sending data: " << e.what() << std::endl;
    }
}

// Parse sensor data from phone app format
// Supports two formats:
// Format 1: "ACCEL:x,y,z|GYRO:x,y,z|ORIENT:x,y,z|MAG:x,y,z"
// Format 2: "timestamp ACC x y z\ntimestamp GYR x y z\n..." (space-separated, one per line)
bool parseSensorData(const std::string& message) {
    try {
        // Debug: print raw message
        static int msgCount = 0;
        if (msgCount++ % 10 == 0) {
            std::cerr << "[DEBUG] Received message: " << message.substr(0, 100) << "..." << std::endl;
        }
        
        std::lock_guard<std::mutex> lock(sensorData.dataMutex);
        
        // Check if this is pipe-separated format (format 1)
        if (message.find('|') != std::string::npos) {
            std::istringstream iss(message);
            std::string token;
            
            while (std::getline(iss, token, '|')) {
                if (token.empty()) continue;
                
                size_t colonPos = token.find(':');
                if (colonPos == std::string::npos) continue;
                
                std::string sensorType = token.substr(0, colonPos);
                std::string values = token.substr(colonPos + 1);
                
                // Parse comma-separated values
                float x, y, z;
                int parsed = sscanf(values.c_str(), "%f,%f,%f", &x, &y, &z);
                if (parsed != 3) continue;
                
                if (sensorType == "ACCEL") {
                    sensorData.accel[0] = x;
                    sensorData.accel[1] = y;
                    sensorData.accel[2] = z;
                }
                else if (sensorType == "GYRO") {
                    sensorData.gyro[0] = x;
                    sensorData.gyro[1] = y;
                    sensorData.gyro[2] = z;
                }
                else if (sensorType == "ORIENT") {
                    sensorData.orientation[0] = x;
                    sensorData.orientation[1] = y;
                    sensorData.orientation[2] = z;
                }
                else if (sensorType == "MAG") {
                    sensorData.magfield[0] = x;
                    sensorData.magfield[1] = y;
                    sensorData.magfield[2] = z;
                }
            }
        }
        else {
            // Format 2: space-separated lines
            std::istringstream iss(message);
            std::string line;
            
            while (std::getline(iss, line)) {
                if (line.empty()) continue;
                
                long timestamp;
                char sensorType[16];
                float x, y, z, w;
                
                // Try to parse the line
                int parsed = sscanf(line.c_str(), "%ld %s %f %f %f %f", &timestamp, sensorType, &x, &y, &z, &w);
                
                if (parsed < 4) continue;
                
                // Convert sensor type abbreviations
                if (strcmp(sensorType, "ACC") == 0) {
                    sensorData.accel[0] = x;
                    sensorData.accel[1] = y;
                    sensorData.accel[2] = z;
                }
                else if (strcmp(sensorType, "GYR") == 0) {
                    sensorData.gyro[0] = x;
                    sensorData.gyro[1] = y;
                    sensorData.gyro[2] = z;
                }
                else if (strcmp(sensorType, "ORI") == 0) {
                    // Orientation can be 3 values (Euler) or 4 values (Quaternion)
                    if (parsed >= 5 && w != 0) {
                        // Has 4 values: quaternion (qx, qy, qz, qw)
                        // Normalize quaternion first
                        float norm = sqrt(x*x + y*y + z*z + w*w);
                        if (norm > 0) {
                            x /= norm;
                            y /= norm;
                            z /= norm;
                            w /= norm;
                        }
                        
                        // Initialize fusion state from first quaternion reading
                        static bool fusionInitialized = false;
                        static int warmupFrames = 0;
                        const int WARMUP_FRAMES = 30;  // Skip sensor fusion for first 30 frames
                        
                        if (!fusionInitialized) {
                            // Use the first incoming quaternion from the phone as initialization
                            // The phone's sensor fusion is already good, so we just adopt its quaternion
                            fusionState.qx = x;
                            fusionState.qy = y;
                            fusionState.qz = z;
                            fusionState.qw = w;
                            normalizeQuaternion(fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw);
                            fusionInitialized = true;
                            warmupFrames = 0;
                            
                            std::cerr << "[INIT] Fusion initialized from first phone quaternion:" << std::endl;
                            std::cerr << "  qx=" << fusionState.qx << " qy=" << fusionState.qy 
                                      << " qz=" << fusionState.qz << " qw=" << fusionState.qw << std::endl;
                            std::cerr << "[INIT] Warmup period: " << WARMUP_FRAMES << " frames (using phone quaternion directly)" << std::endl;
                        }
                        
                        // During warmup, just use incoming quaternion directly to stabilize
                        if (warmupFrames < WARMUP_FRAMES) {
                            fusionState.qx = x;
                            fusionState.qy = y;
                            fusionState.qz = z;
                            fusionState.qw = w;
                            normalizeQuaternion(fusionState.qx, fusionState.qy, fusionState.qz, fusionState.qw);
                            warmupFrames++;
                        } else {
                            // After warmup, apply sensor fusion
                            // Update sensor fusion with current sensor data
                            static auto lastTime = std::chrono::high_resolution_clock::now();
                            auto now = std::chrono::high_resolution_clock::now();
                            float dt = std::chrono::duration<float>(now - lastTime).count();
                            lastTime = now;
                            
                            // Clamp dt to reasonable values (avoid huge jumps on first call)
                            if (dt > 0.1f) dt = 0.1f;
                            if (dt < 0.001f) dt = 0.001f;
                            
                            // Call sensor fusion with accelerometer, gyroscope, and magnetometer data
                            updateSensorFusion(sensorData.accel[0], sensorData.accel[1], sensorData.accel[2],
                                              sensorData.gyro[0], sensorData.gyro[1], sensorData.gyro[2],
                                              sensorData.magfield[0], sensorData.magfield[1], sensorData.magfield[2],
                                              dt);
                        }
                        
                        // Convert fused quaternion to Euler angles (ZYX convention)
                        const float PI = 3.14159265359f;
                        float qx = fusionState.qx;
                        float qy = fusionState.qy;
                        float qz = fusionState.qz;
                        float qw = fusionState.qw;
                        
                        // Roll (rotation around X axis)
                        float sinr_cosp = 2.0f * (qw * qx + qy * qz);
                        float cosr_cosp = 1.0f - 2.0f * (qx * qx + qy * qy);
                        float roll = atan2(sinr_cosp, cosr_cosp) * 180.0f / PI;
                        
                        // Pitch (rotation around Y axis)
                        float sinp = 2.0f * (qw * qy - qz * qx);
                        float pitch;
                        if (fabs(sinp) >= 1.0f)
                            pitch = copysign(90.0f, sinp);
                        else
                            pitch = asin(sinp) * 180.0f / PI;
                        
                        // Yaw (rotation around Z axis)
                        float siny_cosp = 2.0f * (qw * qz + qx * qy);
                        float cosy_cosp = 1.0f - 2.0f * (qy * qy + qz * qz);
                        float yaw = atan2(siny_cosp, cosy_cosp) * 180.0f / PI;
                        
                        // Axis remapping: phone coordinate system to app coordinate system
                        // Direct mapping (no remapping needed)
                        sensorData.orientation[0] = roll;         // App Roll = Phone Roll
                        sensorData.orientation[1] = pitch;        // App Pitch = Phone Pitch
                        sensorData.orientation[2] = yaw;          // App Yaw = Phone Yaw
                        
                        // Copy fused quaternion to sensorData
                        sensorData.fusedQuaternion[0] = qx;
                        sensorData.fusedQuaternion[1] = qy;
                        sensorData.fusedQuaternion[2] = qz;
                        sensorData.fusedQuaternion[3] = qw;
                        
                        static int ori_count = 0;
                        if (ori_count++ % 100 == 0) {
                            std::cerr << "[FUSION] Fused Quat: qx=" << qx << " qy=" << qy << " qz=" << qz << " qw=" << qw << std::endl;
                            std::cerr << "[FUSION] Phone: roll=" << roll << " pitch=" << pitch << " yaw=" << yaw << std::endl;
                            std::cerr << "[FUSION] App: roll=" << sensorData.orientation[0] 
                                      << " pitch=" << sensorData.orientation[1] 
                                      << " yaw=" << sensorData.orientation[2] << std::endl;
                        }
                    } else {
                        // Has 3 values: already Euler angles
                        sensorData.orientation[0] = x;
                        sensorData.orientation[1] = y;
                        sensorData.orientation[2] = z;
                    }
                }
                else if (strcmp(sensorType, "MAG") == 0) {
                    sensorData.magfield[0] = x;
                    sensorData.magfield[1] = y;
                    sensorData.magfield[2] = z;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Parse exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        return false;
    }
}

// Network listener thread
void networkListenerThread(const std::string& bindAddr, uint16_t port) {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return;
    }
#endif

    int serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return;
    }

    // Enable SO_REUSEADDR to allow rapid restarts
    int reuse = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, 
               (const char*)&reuse, sizeof(reuse));
    
    // Set SO_KEEPALIVE for persistent connections
    int keepalive = 1;
    setsockopt(serverSocket, SOL_SOCKET, SO_KEEPALIVE,
               (const char*)&keepalive, sizeof(keepalive));

    struct sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    
    // Parse bind address
    if (bindAddr == "0.0.0.0" || bindAddr.empty()) {
        serverAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (bindAddr == "127.0.0.1" || bindAddr == "localhost") {
        serverAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else {
        // Try to parse as IP address
        if (inet_pton(AF_INET, bindAddr.c_str(), &serverAddr.sin_addr) <= 0) {
            std::cerr << "Invalid bind address: " << bindAddr << std::endl;
            closesocket(serverSocket);
            return;
        }
    }

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "I/O error. Connection unsuccessful" << std::endl;
        std::cerr << "Failed to bind socket to " << bindAddr << ":" << port << std::endl;
        std::perror("bind error");
        closesocket(serverSocket);
        return;
    }

    if (listen(serverSocket, 5) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        closesocket(serverSocket);
        return;
    }

    std::cout << "Listening for sensor data on " << bindAddr << ":" << port << "..." << std::endl;
    std::cout << "Waiting for phone app to connect and stream data..." << std::endl;

    while (running) {
        struct sockaddr_in clientAddr;
        socklen_t clientLen = sizeof(clientAddr);
        
        int clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientLen);
        if (clientSocket < 0) {
            if (running) std::cerr << "Failed to accept client connection" << std::endl;
            continue;
        }

        std::cout << "\n[CONNECTED] Phone app connected from " << inet_ntoa(clientAddr.sin_addr) << std::endl;
        std::cout << "[STATUS] Receiving sensor stream..." << std::endl;

        char buffer[2048];
        int consecutiveEmptyReads = 0;
        const int MAX_EMPTY_READS = 30;
        int totalBytesReceived = 0;

        while (running) {
            std::memset(buffer, 0, sizeof(buffer));
            int bytesReceived = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
            
            if (bytesReceived < 0) {
                std::cerr << "[ERROR] Socket read error" << std::endl;
                break;
            }
            
            if (bytesReceived == 0) {
                consecutiveEmptyReads++;
                if (consecutiveEmptyReads > MAX_EMPTY_READS) {
                    std::cout << "[DISCONNECTED] Phone app stopped streaming (received " << totalBytesReceived << " bytes total)" << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            consecutiveEmptyReads = 0;
            totalBytesReceived += bytesReceived;
            buffer[bytesReceived] = '\0';
            
            std::cerr << "[BYTES] Received " << bytesReceived << " bytes" << std::endl;
            std::cerr << "[RAW] " << buffer << std::endl;
            
            if (parseSensorData(std::string(buffer))) {
                // Mark that we've received data
                {
                    std::lock_guard<std::mutex> lock(dataReceivedMutex);
                    dataReceived = true;
                }
                // Send to GUI visualizer
                sendToGui();
                // Successfully parsed sensor data - continue receiving
            } else {
                std::cerr << "[WARNING] Failed to parse sensor data" << std::endl;
            }
        }

        closesocket(clientSocket);
        std::cout << "[INFO] Connection closed, waiting for next connection..." << std::endl;
        
        // Reset data received flag for next connection
        {
            std::lock_guard<std::mutex> lock(dataReceivedMutex);
            dataReceived = false;
        }
    }

    closesocket(serverSocket);

#ifdef _WIN32
    WSACleanup();
#endif
}

// Display rotation information
void displayRotation() {
    try {
        std::lock_guard<std::mutex> lock(sensorData.dataMutex);
        
        std::cout << "\n========== PHONE ROTATION ==========\n";
        std::cout << std::fixed << std::setprecision(2);
        
        // Display orientation (Roll, Pitch, Yaw)
        std::cout << "Orientation (degrees):\n";
        std::cout << "  Roll:  " << std::setw(8) << sensorData.orientation[0] << "°\n";
        std::cout << "  Pitch: " << std::setw(8) << sensorData.orientation[1] << "°\n";
        std::cout << "  Yaw:   " << std::setw(8) << sensorData.orientation[2] << "°\n";
        
        // Display gyroscope (angular velocity)
        std::cout << "\nAngular Velocity (rad/s):\n";
        std::cout << "  X: " << std::setw(8) << sensorData.gyro[0] << "\n";
        std::cout << "  Y: " << std::setw(8) << sensorData.gyro[1] << "\n";
        std::cout << "  Z: " << std::setw(8) << sensorData.gyro[2] << "\n";
        
        // Display accelerometer
        std::cout << "\nAcceleration (m/s²):\n";
        std::cout << "  X: " << std::setw(8) << sensorData.accel[0] << "\n";
        std::cout << "  Y: " << std::setw(8) << sensorData.accel[1] << "\n";
        std::cout << "  Z: " << std::setw(8) << sensorData.accel[2] << "\n";
        
        // Display magnetic field
        std::cout << "\nMagnetic Field (μT):\n";
        std::cout << "  X: " << std::setw(8) << sensorData.magfield[0] << "\n";
        std::cout << "  Y: " << std::setw(8) << sensorData.magfield[1] << "\n";
        std::cout << "  Z: " << std::setw(8) << sensorData.magfield[2] << "\n";
        
        std::cout << "===================================\n" << std::endl;
    } catch (...) {
        // Silently ignore display errors
    }
}

int main(int argc, char* argv[]) {
    // Default values
    std::string bindAddr = "0.0.0.0";  // Listen on all interfaces by default
    uint16_t port = 3400;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            std::cout << "Phone Sensor Fusion Display Application\n" << std::endl;
            std::cout << "Usage: ./my_app [OPTIONS]\n" << std::endl;
            std::cout << "Options:\n";
            std::cout << "  --bind <addr>     Bind to specific address (default: 0.0.0.0 - all interfaces)\n";
            std::cout << "                    Examples: 0.0.0.0, 127.0.0.1, 192.168.1.100\n";
            std::cout << "  --port <port>     Listen on specific port (default: 3400)\n";
            std::cout << "  --localhost       Bind to localhost (127.0.0.1) only\n";
            std::cout << "  --help            Show this help message\n";
            std::cout << "\nExamples:\n";
            std::cout << "  ./my_app                          # Listen on all interfaces, port 3400\n";
            std::cout << "  ./my_app --port 5000              # Listen on all interfaces, port 5000\n";
            std::cout << "  ./my_app --localhost              # Listen on localhost only, port 3400\n";
            std::cout << "  ./my_app --bind 192.168.1.100 --port 8888\n";
            return 0;
        }
        else if (arg == "--bind" && i + 1 < argc) {
            bindAddr = argv[++i];
        }
        else if (arg == "--port" && i + 1 < argc) {
            try {
                port = static_cast<uint16_t>(std::stoi(argv[++i]));
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << argv[i] << std::endl;
                return 1;
            }
        }
        else if (arg == "--localhost") {
            bindAddr = "127.0.0.1";
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
    }
    
    std::cout << "Phone Sensor Fusion Display Application" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Configuration: " << bindAddr << ":" << port << std::endl;
    std::cout << std::endl;

    // Initialize GUI broadcast
    initGuiBroadcast();

    // Start network listener in a background thread
    std::thread netThread(networkListenerThread, bindAddr, port);
    netThread.detach();

    // Wait for first data connection
    std::cout << "Waiting for phone app to connect and stream data..." << std::endl;
    std::cout << "Press Ctrl+C to exit\n" << std::endl;
    
    bool firstConnection = true;
    int displayCounter = 0;
    
    try {
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Check if we've received data
            {
                std::lock_guard<std::mutex> lock(dataReceivedMutex);
                if (!dataReceived) {
                    // Still waiting for connection
                    continue;
                }
                
                if (firstConnection) {
                    std::cout << "\n[SUCCESS] Phone connected! Displaying rotation data...\n" << std::endl;
                    firstConnection = false;
                }
            }
            
            // Display rotation info every 1 second
            if (++displayCounter % 2 == 0) {
                displayRotation();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in main thread: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in main thread" << std::endl;
    }

    return 0;
}
