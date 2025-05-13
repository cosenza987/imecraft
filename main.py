from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText
import random
import math
import os
import numpy as np

class imeCraft(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.props = WindowProperties()
        self.props.setTitle("instituto militar de engenharia")
        self.props.setSize(1024, 768)
        self.win.requestProperties(self.props)

        self.disableMouse()

        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor((0.4, 0.4, 0.4, 1))
        self.ambient_light_np = self.render.attachNewNode(ambient_light)
        self.render.setLight(self.ambient_light_np)

        directional_light = DirectionalLight("directional_light")
        directional_light.setColor((0.8, 0.8, 0.8, 1))
        directional_light.setDirection(Vec3(-1, -1, -1))
        self.directional_light_np = self.render.attachNewNode(directional_light)
        self.render.setLight(self.directional_light_np)

        print("Loading textures")
        self.textures = self.load_textures()

        self.block_types = {
            0: {'name': 'air', 'texture': None, 'solid': False},
            1: {'name': 'grass', 'texture': self.textures['grass'], 'solid': True},
            2: {'name': 'dirt', 'texture': self.textures['dirt'], 'solid': True},
            3: {'name': 'stone', 'texture': self.textures['stone'], 'solid': True},
        }

        self.world_size = 16   
        self.chunk_size = 16   

        self.chunks = {}
        self.generate_world()

        self.player_pos = Vec3(self.world_size * self.chunk_size // 2, 
                            self.world_size * self.chunk_size // 2, 
                            self.chunk_size + 2)
        self.player_heading = 0
        self.player_pitch = 0
        self.camera.setPos(self.player_pos)

        self.keymap = {
            'forward': False, 'backward': False, 'left': False, 'right': False,
            'up': False, 'down': False, 'place': False, 'remove': False
        }

        self.accept('w', self.update_keymap, ['forward', True])
        self.accept('w-up', self.update_keymap, ['forward', False])
        self.accept('s', self.update_keymap, ['backward', True])
        self.accept('s-up', self.update_keymap, ['backward', False])
        self.accept('a', self.update_keymap, ['left', True])
        self.accept('a-up', self.update_keymap, ['left', False])
        self.accept('d', self.update_keymap, ['right', True])
        self.accept('d-up', self.update_keymap, ['right', False])
        self.accept('space', self.update_keymap, ['up', True])
        self.accept('space-up', self.update_keymap, ['up', False])
        self.accept('shift', self.update_keymap, ['down', True])
        self.accept('shift-up', self.update_keymap, ['down', False])
        self.accept('mouse1', self.update_keymap, ['remove', True])
        self.accept('mouse1-up', self.update_keymap, ['remove', False])
        self.accept('mouse3', self.update_keymap, ['place', True])
        self.accept('mouse3-up', self.update_keymap, ['place', False])

        self.setup_mouse()

        self.crosshair = self.create_crosshair()

        self.taskMgr.add(self.update, "update")
        self.taskMgr.add(self.mouse_task, "mouse_task")

        self.selected_block_type = 1  
        self.accept('1', self.set_selected_block, [1])
        self.accept('2', self.set_selected_block, [2])
        self.accept('3', self.set_selected_block, [3])

        self.last_place_time = 0
        self.place_cooldown = 0.3

        self.setup_block_interaction()

        self.render.setShaderAuto()
        self.setup_lighting()

        self.taskMgr.add(self.update_day_night_cycle, "update_day_night_cycle") 

    def update_day_night_cycle(self, task):
        
        time_of_day = (task.time % 240) / 240.0
        
        angle = time_of_day * 2 * math.pi
        x = math.sin(angle)
        z = -math.cos(angle)
        
        self.sun_light.setDirection(Vec3(x, 0.5, z).normalize())
        
        if 0.25 < time_of_day < 0.75:  # Daytime
            intensity = min(1.0, 1.0 - abs(time_of_day - 0.5) * 2.0)
            self.sun_light.setColor((intensity, intensity * 0.98, intensity * 0.9, 1))
            self.ambient_light.setColor((0.3 * intensity, 0.35 * intensity, 0.45 * intensity, 1))
        else:  # Night time
            night_progress = time_of_day if time_of_day < 0.25 else (time_of_day - 0.75) / 0.25
            night_intensity = 0.15
            self.sun_light.setColor((night_intensity * 0.8, night_intensity * 0.8, night_intensity, 1))
            self.ambient_light.setColor((night_intensity * 0.5, night_intensity * 0.5, night_intensity * 0.8, 1))
        
        return Task.cont

    def setup_lighting(self):
        self.sun_light = DirectionalLight("sun_light")
        self.sun_light.setColor((1.0, 0.98, 0.95, 1))
        self.sun_light.setDirection(Vec3(-0.5, -0.5, -0.7).normalize())
        self.sun_light_np = self.render.attachNewNode(self.sun_light)
        self.render.setLight(self.sun_light_np)
        
        self.ambient_light = AmbientLight("ambient_light")
        self.ambient_light.setColor((0.4, 0.4, 0.45, 1))  
        self.ambient_light_np = self.render.attachNewNode(self.ambient_light)
        self.render.setLight(self.ambient_light_np)
        
        self.fill_light = DirectionalLight("fill_light")
        self.fill_light.setColor((0.3, 0.3, 0.3, 1))  
        self.fill_light.setDirection(Vec3(0.5, 0.5, 0.7).normalize()) 
        self.fill_light_np = self.render.attachNewNode(self.fill_light)
        self.render.setLight(self.fill_light_np)
        
        if hasattr(self, 'sun_light'):
            self.sun_light.setShadowCaster(True, 2048, 2048)
            lens = self.sun_light.getLens()
            lens.setFilmSize(60)
            lens.setNearFar(10, 100)

    def load_textures(self):
        textures = {}
        texture_paths = {
            'grass': 'grass.png',
            'dirt': 'dirt.png',
            'stone': 'stone.png'
        }

        for name, path in texture_paths.items():
            tex = Texture(name)
            if os.path.exists(path):
                print(f"Loading texture: {path}")
                tex = self.loader.loadTexture(path)
            else:
                print(f"Warning: Texture file {path} not found, creating procedural texture.")
                tex = self.create_procedural_texture(name)
            textures[name] = tex
        return textures

    def create_procedural_texture(self, texture_type):

        texture = Texture()
        size = 64
        image = PNMImage(size, size)

        if texture_type == 'grass':

            for y in range(size):
                for x in range(size):
                    noise = random.uniform(-0.1, 0.1)
                    image.setXel(x, y, 0.2 + noise, 0.6 + noise, 0.2 + noise)

        elif texture_type == 'dirt':

            for y in range(size):
                for x in range(size):
                    noise = random.uniform(-0.1, 0.1)
                    image.setXel(x, y, 0.5 + noise, 0.35 + noise, 0.2 + noise)

        elif texture_type == 'stone':

            for y in range(size):
                for x in range(size):
                    noise = random.uniform(-0.1, 0.1)
                    val = 0.5 + noise
                    image.setXel(x, y, val, val, val)

        texture.load(image)
        return texture

    def create_crosshair(self):

        cm = CardMaker("crosshair")
        cm.setFrame(-0.02, 0.02, -0.002, 0.002)  
        horizontal = self.aspect2d.attachNewNode(cm.generate())
        horizontal.setColor(1, 1, 1, 0.5)

        cm = CardMaker("crosshair_vertical")
        cm.setFrame(-0.002, 0.002, -0.02, 0.02)  
        vertical = self.aspect2d.attachNewNode(cm.generate())
        vertical.setColor(1, 1, 1, 0.5)

        crosshair = NodePath("crosshair_parent")
        horizontal.reparentTo(crosshair)
        vertical.reparentTo(crosshair)
        crosshair.reparentTo(self.aspect2d)

        return crosshair

    def setup_block_interaction(self):

        self.interaction_distance = 8.0  

        self.place_cooldown = 0.3  
        self.last_place_time = 0

        self.accept('1', self.set_selected_block, [1])  
        self.accept('2', self.set_selected_block, [2])  
        self.accept('3', self.set_selected_block, [3])  

        self.accept('mouse1', self.remove_block)
        self.accept('mouse3', self.place_block)

        self.show_selected_block_info()

    def show_selected_block_info(self):

        if hasattr(self, 'block_info_text'):
            self.block_info_text.removeNode()

        block_name = self.block_types[self.selected_block_type]['name'].capitalize()
        self.block_info_text = OnscreenText(
            text=f"Selected: {block_name}",
            pos=(-1.3, 0.9),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

    def set_selected_block(self, block_type):
        self.selected_block_type = block_type
        self.show_selected_block_info()
        print(f"Selected block: {self.block_types[block_type]['name']}")

    def remove_block(self):
        result = self.ray_test()
        if result:
            hit_pos, hit_normal, hit_block_pos = result

            block_x, block_y, block_z = int(hit_block_pos.x), int(hit_block_pos.y), int(hit_block_pos.z)
            block_type = self.get_block_at(block_x, block_y, block_z)

            if block_type > 0:

                self.set_block_at(hit_block_pos.x, hit_block_pos.y, hit_block_pos.z, 0)
                print(f"Removed {self.block_types[block_type]['name']} block at ({block_x}, {block_y}, {block_z})")

    def place_block(self):

        current_time = globalClock.getFrameTime()
        if current_time - self.last_place_time < self.place_cooldown:
            return

        self.last_place_time = current_time

        result = self.ray_test()
        if result:
            hit_pos, hit_normal, hit_block_pos = result

            if hit_normal is not None:

                place_pos = hit_block_pos + hit_normal
                place_x, place_y, place_z = int(place_pos.x), int(place_pos.y), int(place_pos.z)

                if self.get_block_at(place_x, place_y, place_z) == 0:

                    player_x, player_y, player_z = int(self.player_pos.x), int(self.player_pos.y), int(self.player_pos.z)
                    player_head_z = int(self.player_pos.z + 1)  

                    if not (place_x == player_x and place_y == player_y and 
                            (place_z == player_z or place_z == player_head_z)):

                        self.set_block_at(place_pos.x, place_pos.y, place_pos.z, self.selected_block_type)
                        block_name = self.block_types[self.selected_block_type]['name']
                        print(f"Placed {block_name} block at ({place_x}, {place_y}, {place_z})")
                    else:
                        print("Can't place block inside player position")
                else:
                    print("Position already occupied by another block")

    def ray_test(self):
        max_distance = self.interaction_distance
        origin = self.camera.getPos()
        direction = self.camera.getMat().getRow3(1).normalized()  

        ray_steps = 200
        step_size = max_distance / ray_steps

        checked_blocks = {}

        for i in range(ray_steps):
            distance = i * step_size
            if distance > max_distance:
                break

            pos = origin + direction * distance
            block_x = int(math.floor(pos.x))
            block_y = int(math.floor(pos.y))
            block_z = int(math.floor(pos.z))

            block_key = (block_x, block_y, block_z)
            if block_key in checked_blocks:
                continue

            checked_blocks[block_key] = True

            block_type = self.get_block_at(block_x, block_y, block_z)

            if block_type in self.block_types and self.block_types[block_type]['solid']:

                prev_pos = origin + direction * (distance - step_size)

                dx = pos.x - prev_pos.x
                dy = pos.y - prev_pos.y
                dz = pos.z - prev_pos.z

                normal = Vec3(0, 0, 0)

                if abs(dx) > abs(dy) and abs(dx) > abs(dz):

                    normal.x = -1 if dx > 0 else 1
                elif abs(dy) > abs(dx) and abs(dy) > abs(dz):

                    normal.y = -1 if dy > 0 else 1
                else:

                    normal.z = -1 if dz > 0 else 1

                return pos, normal, Point3(block_x, block_y, block_z)

        return None

    def set_selected_block(self, block_type):
        self.selected_block_type = block_type
        print(f"Selected block: {self.block_types[block_type]['name']}")

    def update_keymap(self, key, value):
        self.keymap[key] = value

    def mouse_task(self, task):

        if self.mouseWatcherNode.hasMouse():

            props = self.win.getProperties()
            window_center_x = props.getXSize() // 2
            window_center_y = props.getYSize() // 2

            mouse_x = self.mouseWatcherNode.getMouseX()
            mouse_y = self.mouseWatcherNode.getMouseY()

            pixel_x = int((mouse_x + 1.0) * window_center_x)
            pixel_y = int((mouse_y + 1.0) * window_center_y)

            dx = pixel_x - window_center_x
            dy = pixel_y - window_center_y

            self.player_heading -= dx * self.mouse_sensitivity
            self.player_pitch += dy * self.mouse_sensitivity

            self.player_pitch = max(-89, min(89, self.player_pitch))

            self.camera.setHpr(self.player_heading, self.player_pitch, 0)

            self.win.movePointer(0, window_center_x, window_center_y)

        return Task.cont

    def setup_mouse(self):

        self.props.setCursorHidden(True)
        self.props.setMouseMode(WindowProperties.M_relative)
        self.win.requestProperties(self.props)

        self.center_mouse()

        self.mouse_sensitivity = 0.05  

        self.player_heading = 0
        self.player_pitch = 0

    def center_mouse(self):
        props = self.win.getProperties()
        center_x = props.getXSize() // 2
        center_y = props.getYSize() // 2
        self.win.movePointer(0, center_x, center_y)

    def update(self, task):

        dt = globalClock.getDt()

        speed = 5.0 * dt

        movement = Vec3(0, 0, 0)

        forward = self.camera.getMat().getRow3(1)
        forward.z = 0  
        forward.normalize()

        right = self.camera.getMat().getRow3(0)
        right.z = 0  
        right.normalize()

        if self.keymap['forward']:
            movement += forward * speed
        if self.keymap['backward']:
            movement -= forward * speed
        if self.keymap['left']:
            movement -= right * speed
        if self.keymap['right']:
            movement += right * speed
        if self.keymap['up']:
            movement.z += speed
        if self.keymap['down']:
            movement.z -= speed

        new_pos = self.player_pos + movement

        if not self.is_position_solid(new_pos):
            self.player_pos = new_pos
            self.camera.setPos(self.player_pos)

        return Task.cont

    def is_position_solid(self, pos):

        block_x = math.floor(pos.x)
        block_y = math.floor(pos.y)
        block_z = math.floor(pos.z)

        block_type = self.get_block_at(block_x, block_y, block_z)
        return self.block_types[block_type]['solid'] if block_type in self.block_types else False

    def ray_test(self):

        max_distance = 5.0
        origin = self.camera.getPos()
        direction = self.camera.getMat().getRow3(1).normalized()  

        for i in range(100):
            distance = i * 0.05
            if distance > max_distance:
                break

            pos = origin + direction * distance
            block_x = int(math.floor(pos.x))
            block_y = int(math.floor(pos.y))
            block_z = int(math.floor(pos.z))

            block_type = self.get_block_at(block_x, block_y, block_z)

            if block_type in self.block_types and self.block_types[block_type]['solid']:

                prev_pos = origin + direction * (distance - 0.05)

                dx = pos.x - prev_pos.x
                dy = pos.y - prev_pos.y
                dz = pos.z - prev_pos.z

                entry_x = pos.x - block_x
                entry_y = pos.y - block_y
                entry_z = pos.z - block_z

                normal = Vec3(0, 0, 0)

                if abs(dx) > abs(dy) and abs(dx) > abs(dz):

                    normal.x = -1 if dx > 0 else 1
                elif abs(dy) > abs(dx) and abs(dy) > abs(dz):

                    normal.y = -1 if dy > 0 else 1
                else:

                    normal.z = -1 if dz > 0 else 1

                return pos, normal, Point3(block_x, block_y, block_z)

        return None

    def generate_world(self):
        print("Generating world...")

        start_radius = 2  

        center_x = self.world_size // 2
        center_y = self.world_size // 2

        self.generate_chunk(center_x, center_y)
        print(f"Generated central chunk ({center_x},{center_y})")

        for radius in range(1, start_radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                for y in range(center_y - radius, center_y + radius + 1):

                    if (x == center_x - radius or x == center_x + radius or 
                        y == center_y - radius or y == center_y + radius):
                        if 0 <= x < self.world_size and 0 <= y < self.world_size:
                            self.generate_chunk(x, y)
                            print(f"Generated chunk ({x},{y})")

        print("Initial world generation complete!")

    def generate_chunk(self, cx, cy):
        chunk_key = (cx, cy)

        if chunk_key in self.chunks:
            return

        print(f"Generating chunk data for ({cx},{cy})...")

        chunk_data = np.zeros((self.chunk_size, self.chunk_size, self.chunk_size), dtype=np.uint8)

        x_coords = np.arange(self.chunk_size)
        y_coords = np.arange(self.chunk_size)

        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

        global_x = cx * self.chunk_size + X
        global_y = cy * self.chunk_size + Y

        heights = np.floor(8 + 4 * np.sin(global_x * 0.1) * np.cos(global_y * 0.1)).astype(np.int32)

        heights = np.clip(heights, 0, self.chunk_size - 1)

        for x in range(self.chunk_size):
            for y in range(self.chunk_size):
                height = heights[x, y]

                chunk_data[x, y, 0:max(0, height - 3)] = 3

                dirt_start = max(0, height - 3)
                dirt_end = height
                if dirt_end > dirt_start:
                    chunk_data[x, y, dirt_start:dirt_end] = 2

                if 0 <= height < self.chunk_size:
                    chunk_data[x, y, height] = 1

        self.chunks[chunk_key] = {
            'data': chunk_data,
            'node': self.render.attachNewNode(f"chunk_{cx}_{cy}"),
            'meshes': {}
        }

        self.update_chunk_mesh(cx, cy)

    def update_chunk_mesh(self, cx, cy):
        chunk_key = (cx, cy)

        if chunk_key not in self.chunks:
            return

        print(f"Updating mesh for chunk ({cx},{cy})...")
        chunk = self.chunks[chunk_key]

        for block_type in chunk.get('meshes', {}):
            chunk['meshes'][block_type].removeNode()
        chunk['meshes'] = {}

        vertices = []
        indices = []
        uvs = []
        block_types_data = {}  

        for block_type in range(1, len(self.block_types)):  
            start_vertex_idx = len(vertices)
            start_index_idx = len(indices)

            for x in range(self.chunk_size):
                for y in range(self.chunk_size):
                    for z in range(self.chunk_size):
                        if chunk['data'][x, y, z] == block_type:
                            global_x = cx * self.chunk_size + x
                            global_y = cy * self.chunk_size + y
                            global_z = z

                            if global_x + 1 >= cx * self.chunk_size + self.chunk_size or not self.is_block_solid(global_x + 1, global_y, global_z):
                                self.add_face_vertices(vertices, indices, uvs, x, y, z, "right")

                            if global_x - 1 < cx * self.chunk_size or not self.is_block_solid(global_x - 1, global_y, global_z):
                                self.add_face_vertices(vertices, indices, uvs, x, y, z, "left")

                            if global_y + 1 >= cy * self.chunk_size + self.chunk_size or not self.is_block_solid(global_x, global_y + 1, global_z):
                                self.add_face_vertices(vertices, indices, uvs, x, y, z, "front")

                            if global_y - 1 < cy * self.chunk_size or not self.is_block_solid(global_x, global_y - 1, global_z):
                                self.add_face_vertices(vertices, indices, uvs, x, y, z, "back")

                            if global_z + 1 >= self.chunk_size or not self.is_block_solid(global_x, global_y, global_z + 1):
                                self.add_face_vertices(vertices, indices, uvs, x, y, z, "top")

                            if global_z - 1 < 0 or not self.is_block_solid(global_x, global_y, global_z - 1):
                                self.add_face_vertices(vertices, indices, uvs, x, y, z, "bottom")

            end_vertex_idx = len(vertices)
            end_index_idx = len(indices)

            if end_vertex_idx > start_vertex_idx:  
                block_types_data[block_type] = {
                    'start_vertex': start_vertex_idx,
                    'end_vertex': end_vertex_idx,
                    'start_index': start_index_idx,
                    'end_index': end_index_idx
                }

        if len(vertices) > 0:

            vertex_format = GeomVertexFormat.getV3n3t2()
            vertex_data = GeomVertexData('chunk_data', vertex_format, Geom.UHStatic)

            vertex_data.setNumRows(len(vertices))

            vertex_writer = GeomVertexWriter(vertex_data, 'vertex')
            normal_writer = GeomVertexWriter(vertex_data, 'normal')
            texcoord_writer = GeomVertexWriter(vertex_data, 'texcoord')

            normals_map = {
                0: (1, 0, 0), 
                1: (-1, 0, 0),  
                2: (0, 1, 0),   
                3: (0, -1, 0),  
                4: (0, 0, 1),
                5: (0, 0, -1)
            }

            for i in range(len(vertices)):
                vertex_writer.addData3f(*vertices[i])
                face_index = i // 4
                face_type = face_index % 6
                normal_writer.addData3f(*normals_map[face_type])
                
                texcoord_writer.addData2f(*uvs[i])

            for block_type, data in block_types_data.items():

                prim = GeomTriangles(Geom.UHStatic)

                for i in range(data['start_index'], data['end_index']):
                    prim.addVertex(indices[i])

                geom = Geom(vertex_data)
                geom.addPrimitive(prim)

                node = GeomNode(f'block_type_{block_type}')
                node.addGeom(geom)

                np = chunk['node'].attachNewNode(node)
                np.setPos(cx * self.chunk_size, cy * self.chunk_size, 0)
                np.setTwoSided(True)
                np.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
                np.setShaderAuto()
                np.setLightOff()

                texture = self.block_types[block_type]['texture']
                if texture:
                    np.setTexture(texture)

                material = Material()
                material.setAmbient((0.7, 0.7, 0.7, 1))  
                material.setDiffuse((0.9, 0.9, 0.9, 1))   
                material.setEmission((0.1, 0.1, 0.1, 1))  
                material.setShininess(0.0) 
                material.setSpecular((0, 0, 0, 1))  
                np.setMaterial(material)

                chunk['meshes'][block_type] = np

        print(f"Mesh update complete for chunk ({cx},{cy})")

    def add_face_vertices(self, vertices, indices, uvs, x, y, z, face):
        vertex_count = len(vertices)
    
        face_normals = {
            "right":  (1, 0, 0),
            "left":   (-1, 0, 0),
            "front":  (0, 1, 0), 
            "back":   (0, -1, 0),
            "top":    (0, 0, 1),
            "bottom": (0, 0, -1)
        }
        
        normal = face_normals[face]
        
        if face == "right":
            vertices.extend([
                (x + 1, y, z),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1),
                (x + 1, y, z + 1)
            ])
        elif face == "left":
            vertices.extend([
                (x, y, z),
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x, y, z + 1)
            ])
        elif face == "front":
            vertices.extend([
                (x, y + 1, z),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1),
                (x, y + 1, z + 1)
            ])
        elif face == "back":
            vertices.extend([
                (x + 1, y, z),
                (x, y, z),
                (x, y, z + 1),
                (x + 1, y, z + 1)
            ])
        elif face == "top":
            vertices.extend([
                (x, y, z + 1),
                (x + 1, y, z + 1),
                (x + 1, y + 1, z + 1),
                (x, y + 1, z + 1)
            ])
        elif face == "bottom":
            vertices.extend([
                (x, y, z),
                (x + 1, y, z),
                (x + 1, y + 1, z),
                (x, y + 1, z)
            ])

        indices.extend([
            vertex_count, vertex_count + 1, vertex_count + 2,
            vertex_count, vertex_count + 2, vertex_count + 3
        ])

        uvs.extend([
            (0, 0), (1, 0), (1, 1), (0, 1)
        ])

    def is_block_solid(self, x, y, z):
        block_type = self.get_block_at(x, y, z)
        return self.block_types[block_type]['solid'] if block_type in self.block_types else False

    def get_block_at(self, x, y, z):

        cx = x // self.chunk_size
        cy = y // self.chunk_size

        lx = x % self.chunk_size
        ly = y % self.chunk_size

        if (cx, cy) in self.chunks and 0 <= z < self.chunk_size:
            return self.chunks[(cx, cy)]['data'][lx, ly, z]

        return 0

    def set_block_at(self, x, y, z, block_type):

        if not (isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))):
            return

        x, y, z = int(x), int(y), int(z)

        if z < 0:
            print(f"Cannot place block below the world (z={z})")
            return

        max_build_height = 64  
        if z >= max_build_height:
            print(f"Cannot build above height {max_build_height}")
            return

        cx = x // self.chunk_size
        cy = y // self.chunk_size

        lx = x % self.chunk_size
        ly = y % self.chunk_size

        if (cx, cy) not in self.chunks:
            self.generate_chunk(cx, cy)

        chunk = self.chunks[(cx, cy)]
        current_chunk_height = chunk['data'].shape[2]

        if z >= current_chunk_height:

            import numpy as np
            new_height = max(current_chunk_height * 2, z + 1)  
            new_data = np.zeros((self.chunk_size, self.chunk_size, new_height), dtype=np.uint8)
            new_data[:, :, :current_chunk_height] = chunk['data']  
            chunk['data'] = new_data
            print(f"Expanded chunk height from {current_chunk_height} to {new_height}")

        chunk['data'][lx, ly, z] = block_type

        self.update_chunk_mesh(cx, cy)

        if lx == 0 and (cx - 1, cy) in self.chunks:
            self.update_chunk_mesh(cx - 1, cy)
        elif lx == self.chunk_size - 1 and (cx + 1, cy) in self.chunks:
            self.update_chunk_mesh(cx + 1, cy)

        if ly == 0 and (cx, cy - 1) in self.chunks:
            self.update_chunk_mesh(cx, cy - 1)
        elif ly == self.chunk_size - 1 and (cx, cy + 1) in self.chunks:
            self.update_chunk_mesh(cx, cy + 1)

if __name__ == "__main__":
    try:
        print("Starting imeCraft...")
        app = imeCraft()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()