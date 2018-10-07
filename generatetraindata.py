import sc2reader
import os
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL.Image import open as PIL_open
from multiprocessing import Pool

# Set-up
start_time = datetime.now()
replay_folder_path = "replays"
replay_files = os.listdir(replay_folder_path)[4:]

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
    "Barracks",
    "SupplyDepotLowered"
]


class MiniMap:
    def __init__(self, map):
        img = PIL_open(BytesIO(map.minimap))
        img = img.crop(img.getbbox())
        self.map_width, self.map_height = img.size

        offset_x = map.map_info.camera_left
        offset_y = map.map_info.camera_bottom
        map_center = [offset_x + self.map_width / 2.0, offset_y + self.map_height / 2.0]

        self.transX = (self.map_width / 2) + map_center[0]
        self.transY = (self.map_height / 2) + map_center[1]
        self.base = np.zeros(shape=(175, 175, 3), dtype=np.uint8)
        self.base[0: img.height, 0: img.width] = np.array(img)

    def convert_event_coord_to_map_coord(self, x, y):
        img_x = int(self.map_width - self.transX + x)
        img_y = int(self.transY - y)
        return img_x, img_y


def create_training_image(buildings, new_building, count, minimap):
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

    header = str(new_building[0][0]) + "_" + str(new_building[0][1])
    np.save("training/" + header + "_" + str(count) + ".npy", base)


def parse_replay(replay_file):
    path = os.path.join(replay_folder_path, replay_file)
    print(path)
    # Only load enough to check if it's a match we're interested in
    replay = sc2reader.load_replay(path, load_level=1)
    if replay.game_type != "1v1":
        return

    # Load all available info
    replay = sc2reader.load_replay(path, load_level=4, load_map=True)
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
                create_training_image(buildings, b, replay_file, minimap)
                buildings.append(b)
        elif event.name == "UnitDiedEvent" and event.unit.is_building:
            if event.unit.owner.name == my_player.name:
                buildings = [x for x in buildings if x[0] != event.unit.location]


if __name__ == "__main__":
    pool = Pool(8)
    pool.map(parse_replay, replay_files)
    pool.join()
