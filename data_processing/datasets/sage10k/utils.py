from models import Dimensions, Door, Euler, FloorPlan, Object, Point3D, Room, Wall, Window


def dict_to_floor_plan(layout_data: dict) -> FloorPlan:
    """
    Convert a dictionary (from JSON) back to a FloorPlan object.

    Args:
        layout_data: Dictionary containing the floor plan data

    Returns:
        FloorPlan object reconstructed from the dictionary

    Raises:
        ValueError: If the data structure is invalid or incomplete
    """
    try:
        # Convert rooms
        rooms = []
        for room_data in layout_data["rooms"]:
            room = dict_to_room(room_data)
            rooms.append(room)

        # Create FloorPlan object
        floor_plan = FloorPlan(
            id=layout_data["id"],
            rooms=rooms,
            total_area=layout_data["total_area"],
            building_style=layout_data["building_style"],
            description=layout_data["description"],
            created_from_text=layout_data["created_from_text"],
            policy_analysis=layout_data.get("policy_analysis", None)
        )

        return floor_plan

    except KeyError as e:
        raise ValueError(f"Missing required field in layout data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting layout data: {e}")


def dict_to_room(room_data: dict) -> Room:
    """
    Convert a dictionary to a Room object.

    Args:
        room_data: Dictionary containing room data

    Returns:
        Room object reconstructed from the dictionary
    """
    try:
        # Convert position
        position = Point3D(
            x=room_data["position"]["x"],
            y=room_data["position"]["y"],
            z=room_data["position"]["z"]
        )

        # Convert dimensions
        dimensions = Dimensions(
            width=room_data["dimensions"]["width"],
            length=room_data["dimensions"]["length"],
            height=room_data["dimensions"]["height"]
        )

        # Convert walls
        walls = []
        for wall_data in room_data["walls"]:
            wall = dict_to_wall(wall_data)
            walls.append(wall)

        # Convert doors
        doors = []
        for door_data in room_data["doors"]:
            door = dict_to_door(door_data)
            doors.append(door)

        # Convert windows
        windows = []
        for window_data in room_data["windows"]:
            window = dict_to_window(window_data)
            windows.append(window)

        # Convert objects
        objects = []
        for object_data in room_data.get("objects", []):
            obj = dict_to_object(object_data)
            objects.append(obj)

        # Create Room object
        room = Room(
            id=room_data["id"],
            room_type=room_data["room_type"],
            position=position,
            dimensions=dimensions,
            walls=walls,
            doors=doors,
            objects=objects,
            windows=windows,
            floor_material=room_data.get("floor_material", "hardwood"),
            ceiling_height=room_data.get("ceiling_height", 2.7)
        )

        return room

    except KeyError as e:
        raise ValueError(f"Missing required field in room data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting room data: {e}")


def dict_to_wall(wall_data: dict) -> Wall:
    """
    Convert a dictionary to a Wall object.

    Args:
        wall_data: Dictionary containing wall data

    Returns:
        Wall object reconstructed from the dictionary
    """
    try:
        start_point = Point3D(
            x=wall_data["start_point"]["x"],
            y=wall_data["start_point"]["y"],
            z=wall_data["start_point"]["z"]
        )

        end_point = Point3D(
            x=wall_data["end_point"]["x"],
            y=wall_data["end_point"]["y"],
            z=wall_data["end_point"]["z"]
        )

        wall = Wall(
            id=wall_data["id"],
            start_point=start_point,
            end_point=end_point,
            height=wall_data["height"],
            thickness=wall_data.get("thickness", 0.1),
            material=wall_data.get("material", "drywall")
        )

        return wall

    except KeyError as e:
        raise ValueError(f"Missing required field in wall data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting wall data: {e}")


def dict_to_door(door_data: dict) -> Door:
    """
    Convert a dictionary to a Door object.

    Args:
        door_data: Dictionary containing door data

    Returns:
        Door object reconstructed from the dictionary
    """
    try:
        door = Door(
            id=door_data["id"],
            wall_id=door_data["wall_id"],
            position_on_wall=door_data["position_on_wall"],
            width=door_data["width"],
            height=door_data["height"],
            door_type=door_data.get("door_type", "standard"),
            opens_inward=door_data.get("opens_inward", True),
            opening=door_data.get("opening", False),  # Handle opening property
            door_material=door_data.get("door_material", "wood")
        )

        return door

    except KeyError as e:
        raise ValueError(f"Missing required field in door data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting door data: {e}")


def dict_to_window(window_data: dict) -> Window:
    """
    Convert a dictionary to a Window object.

    Args:
        window_data: Dictionary containing window data

    Returns:
        Window object reconstructed from the dictionary
    """
    try:
        window = Window(
            id=window_data["id"],
            wall_id=window_data["wall_id"],
            position_on_wall=window_data["position_on_wall"],
            width=window_data["width"],
            height=window_data["height"],
            sill_height=window_data["sill_height"],
            window_type=window_data.get("window_type", "standard"),
            window_material=window_data.get("window_material", "standard")
        )

        return window

    except KeyError as e:
        raise ValueError(f"Missing required field in window data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting window data: {e}")


def dict_to_object(object_data: dict) -> Object:
    """
    Convert a dictionary to an Object object.

    Args:
        object_data: Dictionary containing object data

    Returns:
        Object object reconstructed from the dictionary
    """
    try:
        # Convert position
        position = Point3D(
            x=object_data["position"]["x"],
            y=object_data["position"]["y"],
            z=object_data["position"]["z"]
        )

        # Convert rotation
        rotation = Euler(
            x=object_data["rotation"]["x"],
            y=object_data["rotation"]["y"],
            z=object_data["rotation"]["z"]
        )

        # Convert dimensions
        dimensions = Dimensions(
            width=object_data["dimensions"]["width"],
            length=object_data["dimensions"]["length"],
            height=object_data["dimensions"]["height"]
        )

        obj = Object(
            id=object_data["id"],
            room_id=object_data["room_id"],
            type=object_data["type"],
            description=object_data["description"],
            position=position,
            rotation=rotation,
            dimensions=dimensions,
            source=object_data["source"],
            source_id=object_data["source_id"],
            place_id=object_data["place_id"],
            mass=object_data.get("mass", 1.0),
            placement_constraints=object_data.get("placement_constraints", None)
        )

        return obj

    except KeyError as e:
        raise ValueError(f"Missing required field in object data: {e}")
    except Exception as e:
        raise ValueError(f"Error converting object data: {e}")
