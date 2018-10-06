import sc2reader
import os
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL.Image import open as PIL_open, ANTIALIAS
import matplotlib.pyplot as plt

from PIL import Image
import cv2

count = 0
# Set-up
start_time = datetime.now()
replay_folder_path = "replays"
replay_files = os.listdir(replay_folder_path)[4:]
game_screen_width = 200
game_screen_height = 200

# GUI
player_1_name = ""
player_2_name = ""

# minimap stuff
minimap = None

unique = [
    "Refinery"
    "SupplyDepotLowered",
    "BarracksReactor",
    "FactoryFlying",
    "SupplyDepot",
    "BarracksTechLab",
    "OrbitalCommand",
    "BarracksTechLab",
    "EngineeringBay",
    "Bunker",
    "StarportReactor",
    "Starport",
    "StarportTechLab",
    "FusionCore",
    "MissileTurret",
    "Factory",
    "FactoryReactor",
    "Armory",
    "BarracksFlying",
    "TechLab",
    "OrbitalCommandFlying",
    "FactoryTechLab",
    "SensorTower",
    "CommandCenterFlying",
    "CommandCenter",
    "GhostAcademy",
    "PlanetaryFortress",
    "Reactor",
    "Barracks"
]

class MiniMap:
    def __init__(self, map):
        self.map = map
        self.width = map.map_info.width
        self.height = map.map_info.height

        img = PIL_open(BytesIO(map.minimap))
        img = img.crop(img.getbbox())
        cropsize = img.size

        self.map_width = cropsize[0]
        self.map_height = cropsize[1]

        self.mapSize = self.map_height * self.map_width
        offset_x = map.map_info.camera_left
        offset_y = map.map_info.camera_bottom
        map_center = [offset_x + cropsize[0] / 2.0, offset_y + cropsize[1] / 2.0]

        # this is the center of the minimap image, in pixel coordinates
        imageCenter = [(self.map_width / 2), self.map_height / 2]
        self.transX = imageCenter[0] + map_center[0]
        self.transY = imageCenter[1] + map_center[1]
        self.base = np.zeros(shape=(228, 228, 3), dtype=np.uint8)
        self.base[0: img.height, 0: img.width] = np.array(img)

    def convert_event_coord_to_map_coord(self, x, y):
        img_x = int(self.map_width - self.transX + x)
        img_y = int(self.transY - y)
        return img_x, img_y

def create_training_image(buildings, new_building, count):
    base = minimap.base.copy()

    if new_building[1] not in unique:
        unique.append(new_building[1])

    for building in buildings:
        building_id = unique.index(building[1])
        x, y = building[0]
        y, x = minimap.convert_event_coord_to_map_coord(x, y)

        base[x][y][0] = 0
        base[x][y][1] = 0
        base[x][y][2] = 255 - (building_id*6)

    img = Image.fromarray(base)
    img = img.resize((700, 700), ANTIALIAS)
    cv2.imshow('image', np.array(img))
    cv2.waitKey(2)
    header = str(new_building[1]) + "\n" + str(new_building[0])
    np.savetxt("training/test_" + str(count) + ".txt", base.reshape((228*3, 228)), fmt='%i', delimiter=",", header=header)

plt.ioff()
for replay_file in replay_files:
    path = os.path.join(replay_folder_path, replay_file)
    print(path)
    # Only load enough to check if it's a match we're interested in
    replay = sc2reader.load_replay(path, load_level=1)
    if replay.game_type != "1v1":
        continue

    if replay.attributes[1]["Race"] != "Terran" and replay.attributes[2]["Race"] != "Terran":
        continue

    # Load all available info
    replay = sc2reader.load_replay(path, load_level=4, load_map=True)

    player_1_name, player_2_name = [x.name for x in replay.players]
    my_player = None
    minimap = MiniMap(replay.map)

    is_terran = [replay.players[0].play_race == "Terran", replay.players[1].play_race == "Terran"]
    if sum(is_terran) == 1:
        # if only one Terran player (gets the index of the first true value in the list)
        my_player = replay.players[[i for i, x in enumerate(is_terran) if x][0]]
    else:
        # if two Terran players we choose the winning one
        my_player = replay.winner.players[0]

    buildings = []
    for event in replay.events:
        if event.name == "UnitDoneEvent" and event.unit.is_building:
            if event.unit.owner.name == my_player.name:
                b = (event.unit.location, event.unit.title, event.second)
                create_training_image(buildings, b, count)
                buildings.append(b)
                count += 1
        elif event.name == "UnitDiedEvent" and event.unit.is_building:
            if event.unit.owner.name == my_player.name:
                buildings = [x for x in buildings if x[0] != event.unit.location]

print(datetime.now() - start_time)
