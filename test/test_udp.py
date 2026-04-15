import socket

# --- CONFIGURATION ---
ESP32_IP = "10.64.50.62"  # Your exact ESP32 IP
UDP_PORT = 4210              # The port we set in the C++ code

print("=========================================")
print(f" 🚀 Mac-to-ESP32 UDP Test Transmitter")
print(f" Target: {ESP32_IP} on port {UDP_PORT}")
print("=========================================")
print("Instructions:")
print(" - Type '180,180' to simulate driving forward")
print(" - Type '200,0' to simulate a sharp right turn")
print(" - Type 'PING' to send a heartbeat")
print(" - Type 'q' to quit")
print("=========================================\n")

# Create the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    try:
        # Get your input from the terminal
        cmd = input("Enter command: ")
        
        if cmd.lower() == 'q':
            break
            
        # The C++ code looks for a newline character at the end
        payload = f"{cmd}\n".encode('utf-8')
        
        # Fire it over Wi-Fi
        sock.sendto(payload, (ESP32_IP, UDP_PORT))
        print(f"  ✅ Packet '{cmd}' sent through the air!")
        
    except KeyboardInterrupt:
        break

sock.close()
print("\nTransmitter shut down.")