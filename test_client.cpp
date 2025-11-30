#include <iostream>
#include <cstring>

#ifdef _WIN32
    #include <winsock2.h>
    #pragma comment(lib, "ws2_32.lib")
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #define closesocket close
#endif

int main() {
    std::cout << "Test client connecting to 127.0.0.1:3400..." << std::endl;
    
    int clientSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (clientSocket < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        return 1;
    }

    struct sockaddr_in serverAddr;
    std::memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(3400);
    serverAddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(clientSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        std::cerr << "Failed to connect to server" << std::endl;
        std::perror("connect");
        closesocket(clientSocket);
        return 1;
    }

    std::cout << "Connected! Sending test sensor data..." << std::endl;

    const char* testData = "ACCEL:1.0,2.0,9.8|GYRO:0.01,0.02,-0.01|ORIENT:5.3,2.1,45.6|MAG:25.5,-5.3,50.1";
    
    for (int i = 0; i < 10; i++) {
        if (send(clientSocket, testData, strlen(testData), 0) < 0) {
            std::cerr << "Failed to send data" << std::endl;
            break;
        }
        std::cout << "Sent packet " << (i+1) << std::endl;
        usleep(500000);  // 500ms delay
    }

    closesocket(clientSocket);
    std::cout << "Test complete!" << std::endl;
    return 0;
}
