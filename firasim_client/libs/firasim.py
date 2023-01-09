import socket
import pathlib
moduleFolder = str(pathlib.Path(__file__).parent.absolute())
import sys
sys.path.append(moduleFolder + '/protobuf/')
sys.path.append(moduleFolder + '/../')
from firasim_client.libs import packet_pb2
from firasim_client.libs import constants
import time

class FIRASimVision:
    def __init__(self, host=constants.HOST_FIRASIM_VISION, port=constants.PORT_FIRASIM_VISION):
        self.host = host
        self.port = port
        self.socket = self.createSocket(host, port)

    def createSocket(self, host, port, blocking = False):
        # create UDP c  
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(blocking)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32) 
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        sock.bind((host, port))
        
        selfHost = socket.gethostbyname(socket.gethostname())
        sock.setsockopt(socket.SOL_IP, socket.IP_MULTICAST_IF, socket.inet_aton(selfHost))
        sock.setsockopt(socket.SOL_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(host) + socket.inet_aton(selfHost))

        return sock

    def read(self):
        try:
            data = self.socket.recv(512)
            
            if len(data) > 0:
                environment = packet_pb2.Environment()
                environment.ParseFromString(data)
                return environment
            return None        
        except:
            return None

class FIRASimCommand:
    def __init__(self, host=constants.HOST_FIRASIM_COMMAND, port=constants.PORT_FIRASIM_COMMAND, team_yellow = False):
        self.host = host
        self.port = port
        self.team_yellow = team_yellow

        self.socket = self.createSocket(host, port)

    def createSocket(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((host, port))

        return sock

    def write(self, index, vl, vr):
        packet = packet_pb2.Packet()
        command = packet.cmd.robot_commands.add()

        command.yellowteam = self.team_yellow
        command.id = index
        command.wheel_left = vl
        command.wheel_right = vr

        self.socket.send(packet.SerializeToString())

    def writeMulti(self, actions):
        packet = packet_pb2.Packet()

        for i, (vl, vr) in enumerate(actions):
            command = packet.cmd.robot_commands.add()

            command.yellowteam = self.team_yellow
            command.id = i
            command.wheel_left = vl
            command.wheel_right = vr
        
        self.socket.send(packet.SerializeToString())

    def setPos(self, index, x, y, th):
        packet = packet_pb2.Packet()
        robotReplacement = packet.replace.robots.add()
        robot = robotReplacement.position

        robotReplacement.yellowteam = self.team_yellow
        robotReplacement.turnon = True

        robot.robot_id = index
        robot.x = x
        robot.y = y
        robot.orientation = th * 360
        robot.vx = 0
        robot.vy = 0
        robot.vorientation = 0

        self.socket.send(packet.SerializeToString())
        

    def setBallPos(self, x, y):
        packet = packet_pb2.Packet()
        ballReplacement = packet.replace.ball

        ballReplacement.x = x
        ballReplacement.y = y

        self.socket.send(packet.SerializeToString())

if __name__ == "__main__":
    command = FIRASimCommand()
    vision = FIRASimVision()

    command.setBallPos(0,0)

    t0 = time.time()
    while True:
        packet = vision.read()
        if packet is not None:
            t1 = time.time()
            print(packet)
            print((t1 - t0)*1000)
            t0 = t1
        # command.write(0, 10, -10)
        command.setPos(1, 0, 0, 0)
        # time.sleep(0.01)
