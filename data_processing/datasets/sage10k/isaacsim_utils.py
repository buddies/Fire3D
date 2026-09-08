import numpy as np
import trimesh
import sys
import json
import os
from isaacsim import SimulationApp
simulation_app = None

def start_simulation_app():

    global simulation_app
    simulation_app = SimulationApp({"headless": True})
    print("Starting simulation app...")


from tex_utils_local import (
    dict_to_floor_plan,
    export_layout_to_mesh_dict_list_v2,
    export_layout_to_mesh_dict_list_no_object_transform_v2
)

def AddTranslate(top, offset):
    top.AddTranslateOp().Set(value=offset)

def convert_mesh_to_usd(stage, usd_internal_path, verts, faces, collision_approximation, static, articulation,
                        physics_iter=(16, 1), mass=None, apply_debug_torque=False, debug_torque_value=50.0,
                        texture=None, usd_internal_art_reference_path="/World",
                        add_damping=False):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]

    points = verts

    # bbox_max = np.max(points, axis=0)
    # bbox_min = np.min(points, axis=0)
    # center = (bbox_max + bbox_min) / 2
    # points = points - center
    # center = (center[0], center[1], center[2])

    vertex_counts = np.ones(n_faces).astype(np.int32) * 3

    mesh = UsdGeom.Mesh.Define(stage, usd_internal_path)

    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    # mesh.CreateDisplayColorPrimvar("vertex")
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(vertex_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])

    # tilt = mesh.AddRotateXOp(opSuffix='tilt')
    # tilt.Set(value=-90)
    # AddTranslate(mesh, center)

    prim = stage.GetPrimAtPath(usd_internal_path)

    if texture is not None:
        vts = texture["vts"]
        fts = texture["fts"]
        texture_map_path = texture["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",
                                    Sdf.ValueTypeNames.TexCoord2fArray,
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))

        usd_mat_path = usd_internal_path+"_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        if "pbr_parameters" in texture:
            pbr_parameters = texture["pbr_parameters"]
            roughness = pbr_parameters.get("roughness", 1.0)
            metallic = pbr_parameters.get("metallic", 0.0)
        else:
            roughness = 1.0
            metallic = 0.0
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        # Note here we are connecting the shader's input to the material's
        # "public interface" attribute. This allows users to change the primvar name
        # on the material itself without drilling inside to examine shader nodes.
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Now bind the Material to the card
        mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(mesh).Bind(material)


    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    if not static:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if mass is not None:
            mass_api.CreateMassAttr(mass)
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        ps_rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physx_rigid_body.CreateSolverPositionIterationCountAttr(physics_iter[0])
        physx_rigid_body.CreateSolverVelocityIterationCountAttr(physics_iter[1])

    if articulation is not None:
        articulation_api = UsdPhysics.ArticulationRootAPI.Apply(prim)
        # Add revolute joint articulation
        rotate_axis_point_lower, rotate_axis_point_upper = articulation

        # Calculate the rotation axis vector
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize
        # Create a revolute joint
        joint_path = usd_internal_path + "_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        # Set the joint axis (in local space)
        joint.CreateAxisAttr("Z")  # Default to Z-axis, we'll transform to match our axis
        # Set the joint bodies - this connects the joint to the rigid body
        joint.CreateBody0Rel().SetTargets([usd_internal_path])
        joint.CreateBody1Rel().SetTargets([usd_internal_art_reference_path])
        # For a single body joint (attached to world), we don't set body1
        # Create joint position (midpoint of the axis)
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2
        # Apply transform to position the joint at the rotation axis
        joint_prim = stage.GetPrimAtPath(joint_path)
        joint_xform = UsdGeom.Xformable(joint_prim)
        # Set the joint position using physics:localPos0 and physics:localPos1
        # These define the connection points on each body
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))

        # # Also set the transform position for visualization/debugging
        # translate_op = joint_xform.AddTranslateOp()
        # translate_op.Set(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        # If the rotation axis is not along Z, we need to rotate the joint
        if not np.allclose(axis_vector, [0, 0, 1]):
            # Calculate rotation to align Z-axis with our desired axis
            z_axis = np.array([0, 0, 1])
            # Use cross product to find rotation axis
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-6:  # Not parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # Calculate angle between vectors
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))
                # Convert to degrees
                angle_degrees = np.degrees(angle)

                # Create rotation quaternion
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0],
                               sin_half * rotation_axis[1], sin_half * rotation_axis[2])

                # Apply rotation using physics:localRot0 and physics:localRot1
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)
        # Optional: Set joint limits if needed
        # joint.CreateLowerLimitAttr(-180.0)  # -180 degrees
        # joint.CreateUpperLimitAttr(180.0)   # +180 degrees

        # Apply debug torque if requested (for testing joint functionality)
        if apply_debug_torque:
            print(f"Applying debug torque: {debug_torque_value}")
            # Apply DriveAPI to the joint
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            # Set drive type to velocity control
            drive_api.CreateTypeAttr("force")
            # Set target velocity to make the joint rotate
            drive_api.CreateTargetVelocityAttr(debug_torque_value)  # degrees per second
            # Set drive stiffness and damping
            drive_api.CreateStiffnessAttr(0.0)  # No position control
            drive_api.CreateDampingAttr(1e4)   # High damping for velocity control
            # Set max force
            drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive
            print("Debug torque applied - joint should rotate")

        # Apply PhysX-specific joint properties for better simulation
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        # Note: Break force/torque attributes may not be available for all joint types
        # physx_joint.CreateBreakForceAttr(1e10)  # Very large value - effectively never break
        # physx_joint.CreateBreakTorqueAttr(1e10)  # Very large value - effectively never break
    UsdPhysics.CollisionAPI.Apply(prim)
    ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    ps_collision_api.CreateContactOffsetAttr(0.005)
    ps_collision_api.CreateRestOffsetAttr(0.001)
    ps_collision_api.CreateTorsionalPatchRadiusAttr(0.01)

    physx_rigid_body.CreateLinearDampingAttr(10.0)
    physx_rigid_body.CreateAngularDampingAttr(10.0)

    physx_rigid_body.CreateMaxLinearVelocityAttr(0.5)
    physx_rigid_body.CreateMaxAngularVelocityAttr(0.5)
    physx_rigid_body.CreateMaxDepenetrationVelocityAttr(50.0)

    # physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
    # physxSceneAPI.CreateGpuTempBufferCapacityAttr(16 * 1024 * 1024 * 2)
    # physxSceneAPI.CreateGpuHeapCapacityAttr(64 * 1024 * 1024 * 2)

    if collision_approximation == "sdf":
        physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        physx_sdf.CreateSdfResolutionAttr(256)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("sdf")
    elif collision_approximation == "convexDecomposition":
        convexdecomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        collider = UsdPhysics.MeshCollisionAPI.Apply(prim)
        collider.CreateApproximationAttr("convexDecomposition")

    mat = UsdPhysics.MaterialAPI.Apply(prim)
    mat.CreateDynamicFrictionAttr(1e20)
    mat.CreateStaticFrictionAttr(1e20)
    # mat.CreateDynamicFrictionAttr(2.0)  # Increased from 0.4 for better grasping
    # mat.CreateStaticFrictionAttr(2.0)   # Increased from 0.4 for better grasping

    return stage


def door_frame_to_usd(
    stage,
    usd_internal_path_door,
    usd_internal_path_door_frame,
    mesh_obj_door,
    mesh_obj_door_frame,
    articulation_door,
    texture_door,
    texture_door_frame,
    apply_debug_torque=False,
    debug_torque_value=50.0
):
    """
    Create door and door frame USD objects with a revolute joint between them.

    Args:
        stage: USD stage
        usd_internal_path_door: USD path for the door
        usd_internal_path_door_frame: USD path for the door frame
        mesh_obj_door: Trimesh object for the door
        mesh_obj_door_frame: Trimesh object for the door frame
        articulation_door: Tuple of (rotate_axis_point_lower, rotate_axis_point_upper) in world coordinates
        texture_door: Texture info for door (can be None)
        texture_door_frame: Texture info for door frame (can be None)
        apply_debug_torque: Whether to apply debug torque for testing joint functionality
        debug_torque_value: Target velocity for debug torque (degrees per second)
    """
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade

    # Extract vertices and faces from mesh objects
    door_verts = np.array(mesh_obj_door.vertices)
    door_faces = np.array(mesh_obj_door.faces)
    frame_verts = np.array(mesh_obj_door_frame.vertices)
    frame_faces = np.array(mesh_obj_door_frame.faces)

    # Create the door frame first (this will be the static parent)
    # Door frame is static and acts as the base for the joint
    n_frame_verts = frame_verts.shape[0]
    n_frame_faces = frame_faces.shape[0]
    frame_vertex_counts = np.ones(n_frame_faces).astype(np.int32) * 3

    # Create door frame mesh
    frame_mesh = UsdGeom.Mesh.Define(stage, usd_internal_path_door_frame)
    frame_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(frame_verts))
    frame_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(frame_vertex_counts))
    frame_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(frame_faces))
    frame_mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])

    frame_prim = stage.GetPrimAtPath(usd_internal_path_door_frame)

    # Apply texture to door frame if provided
    if texture_door_frame is not None:
        vts = texture_door_frame["vts"]
        fts = texture_door_frame["fts"]
        texture_map_path = texture_door_frame["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(frame_mesh).CreatePrimvar("st",
                                    Sdf.ValueTypeNames.TexCoord2fArray,
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))

        usd_mat_path = usd_internal_path_door_frame + "_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Bind material to door frame
        frame_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(frame_mesh).Bind(material)

    # Set up door frame physics (static)
    # UsdPhysics.CollisionAPI.Apply(frame_prim)
    # UsdPhysics.RigidBodyAPI.Apply(frame_prim)

    # Apply physics material to door frame
    frame_mat = UsdPhysics.MaterialAPI.Apply(frame_prim)
    frame_mat.CreateDynamicFrictionAttr(2.0)
    frame_mat.CreateStaticFrictionAttr(2.0)

    # Create the door (this will be the moving part)
    n_door_verts = door_verts.shape[0]
    n_door_faces = door_faces.shape[0]
    door_vertex_counts = np.ones(n_door_faces).astype(np.int32) * 3

    # Create door mesh
    door_mesh = UsdGeom.Mesh.Define(stage, usd_internal_path_door)
    door_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(door_verts))
    door_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(door_vertex_counts))
    door_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(door_faces))
    door_mesh.CreateExtentAttr([(-100, -100, -100), (100, 100, 100)])

    door_prim = stage.GetPrimAtPath(usd_internal_path_door)

    # Apply texture to door if provided
    if texture_door is not None:
        vts = texture_door["vts"]
        fts = texture_door["fts"]
        texture_map_path = texture_door["texture_map_path"]
        tex_coords = vts[fts.reshape(-1)].reshape(-1, 2)

        texCoords = UsdGeom.PrimvarsAPI(door_mesh).CreatePrimvar("st",
                                    Sdf.ValueTypeNames.TexCoord2fArray,
                                    UsdGeom.Tokens.faceVarying)
        texCoords.Set(Vt.Vec2fArray.FromNumpy(tex_coords))

        usd_mat_path = usd_internal_path_door + "_mat"
        material = UsdShade.Material.Define(stage, usd_mat_path)
        stInput = material.CreateInput('frame:stPrimvarName', Sdf.ValueTypeNames.Token)
        stInput.Set('st')

        pbrShader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/PBRShader")
        pbrShader.CreateIdAttr("UsdPreviewSurface")
        pbrShader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        pbrShader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        pbrShader.CreateInput('useSpecularWorkflow', Sdf.ValueTypeNames.Bool).Set(True)

        material.CreateSurfaceOutput().ConnectToSource(pbrShader.ConnectableAPI(), "surface")

        # create texture coordinate reader
        stReader = UsdShade.Shader.Define(stage, f"{usd_mat_path}/stReader")
        stReader.CreateIdAttr('UsdPrimvarReader_float2')
        stReader.CreateInput('varname',Sdf.ValueTypeNames.Token).ConnectToSource(stInput)

        # diffuse texture
        diffuseTextureSampler = UsdShade.Shader.Define(stage, f"{usd_mat_path}/diffuseTexture")
        diffuseTextureSampler.CreateIdAttr('UsdUVTexture')
        diffuseTextureSampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(texture_map_path)
        diffuseTextureSampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stReader.ConnectableAPI(), 'result')
        diffuseTextureSampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        pbrShader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(diffuseTextureSampler.ConnectableAPI(), 'rgb')

        # Bind material to door
        door_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(door_mesh).Bind(material)

    # Set up door physics (dynamic)
    UsdPhysics.CollisionAPI.Apply(door_prim)
    mass_api = UsdPhysics.MassAPI.Apply(door_prim)
    mass_api.CreateMassAttr(10.0)  # Set door mass to 10kg
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(door_prim)

    # Apply PhysX rigid body properties for better simulation
    physx_rigid_body = PhysxSchema.PhysxRigidBodyAPI.Apply(door_prim)
    physx_rigid_body.CreateSolverPositionIterationCountAttr(255)
    physx_rigid_body.CreateSolverVelocityIterationCountAttr(255)
    physx_rigid_body.CreateLinearDampingAttr(10.0)
    physx_rigid_body.CreateAngularDampingAttr(10.0)
    physx_rigid_body.CreateMaxLinearVelocityAttr(0.5)
    physx_rigid_body.CreateMaxAngularVelocityAttr(0.5)
    physx_rigid_body.CreateMaxDepenetrationVelocityAttr(50.0)

    # # Apply collision properties
    # ps_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(door_prim)
    # ps_collision_api.CreateContactOffsetAttr(0.005)
    # ps_collision_api.CreateRestOffsetAttr(0.001)
    # ps_collision_api.CreateTorsionalPatchRadiusAttr(0.01)

    physx_sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(door_prim)
    physx_sdf.CreateSdfResolutionAttr(256)
    collider = UsdPhysics.MeshCollisionAPI.Apply(door_prim)
    collider.CreateApproximationAttr("sdf")

    # Apply physics material to door
    door_mat = UsdPhysics.MaterialAPI.Apply(door_prim)
    door_mat.CreateDynamicFrictionAttr(2.0)
    door_mat.CreateStaticFrictionAttr(2.0)

    # Create the revolute joint between door and door frame
    if articulation_door is not None:
        rotate_axis_point_lower, rotate_axis_point_upper = articulation_door

        # Calculate the rotation axis vector
        axis_vector = np.array(rotate_axis_point_upper) - np.array(rotate_axis_point_lower)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Normalize

        # Create a revolute joint
        joint_path = usd_internal_path_door + "_hinge_joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)

        # Set the joint axis (in local space)
        joint.CreateAxisAttr("Z")  # Default to Z-axis, we'll transform to match our axis

        # Set the joint bodies - door rotates relative to door frame
        joint.CreateBody0Rel().SetTargets([usd_internal_path_door])      # Moving body (door)
        joint.CreateBody1Rel().SetTargets([usd_internal_path_door_frame]) # Static body (door frame)

        # Create joint position (midpoint of the axis)
        joint_pos = (np.array(rotate_axis_point_lower) + np.array(rotate_axis_point_upper)) / 2

        # Set the joint position using physics:localPos0 and physics:localPos1
        # These define the connection points on each body
        joint.CreateLocalPos0Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))
        joint.CreateLocalPos1Attr(Gf.Vec3f(joint_pos[0], joint_pos[1], joint_pos[2]))

        # If the rotation axis is not along Z, we need to rotate the joint
        if not np.allclose(axis_vector, [0, 0, 1]):
            # Calculate rotation to align Z-axis with our desired axis
            z_axis = np.array([0, 0, 1])
            # Use cross product to find rotation axis
            rotation_axis = np.cross(z_axis, axis_vector)
            if np.linalg.norm(rotation_axis) > 1e-6:  # Not parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                # Calculate angle between vectors
                angle = np.arccos(np.clip(np.dot(z_axis, axis_vector), -1.0, 1.0))

                # Create rotation quaternion
                sin_half = np.sin(angle / 2)
                cos_half = np.cos(angle / 2)
                quat = Gf.Quatf(cos_half, sin_half * rotation_axis[0],
                               sin_half * rotation_axis[1], sin_half * rotation_axis[2])

                # Apply rotation using physics:localRot0 and physics:localRot1
                joint.CreateLocalRot0Attr(quat)
                joint.CreateLocalRot1Attr(quat)

        # Set joint limits for a typical door (0 to 120 degrees)
        joint.CreateLowerLimitAttr(-120.0)    # 0 degrees (closed)
        joint.CreateUpperLimitAttr(120.0)  # 120 degrees (open)

        # Apply PhysX-specific joint properties for better simulation
        joint_prim = stage.GetPrimAtPath(joint_path)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)

        # Apply debug torque if requested (for testing joint functionality)
        if apply_debug_torque:
            print(f"Applying debug torque to door joint: {debug_torque_value}")
            # Apply DriveAPI to the joint
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            # Set drive type to velocity control
            drive_api.CreateTypeAttr("force")
            # Set target velocity to make the door rotate
            drive_api.CreateTargetVelocityAttr(debug_torque_value)  # degrees per second
            # Set drive stiffness and damping
            drive_api.CreateStiffnessAttr(0.0)  # No position control
            drive_api.CreateDampingAttr(100)   # High damping for velocity control
            # Set max force
            drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive
            print("Debug torque applied - door should rotate")
        else:
            # Add some damping to make the door movement more realistic without active torque
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive_api.CreateTypeAttr("force")
            drive_api.CreateStiffnessAttr(0.0)  # No position control
            drive_api.CreateDampingAttr(100.0)  # Add damping for realistic movement
            drive_api.CreateMaxForceAttr(1000.0)  # Maximum force for the drive

    return stage


def save_usd_with_ids(usd_file_path, mesh_info_dict, room_base_ids):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade


    stage = Usd.Stage.CreateNew(usd_file_path)

    collision_approximation = "sdf"
    # collision_approximation = "convexDecomposition"


    world_base_prim = UsdGeom.Xform.Define(stage, "/World")

    # set default prim to World
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    for mesh_id in room_base_ids:
        if mesh_id.startswith("door_"):
            continue
        else:
            usd_internal_path = f"/World/{mesh_id}"
        mesh_dict = mesh_info_dict[mesh_id]
        mesh_obj_i = mesh_dict['mesh']
        static = mesh_dict['static']
        articulation = mesh_dict.get('articulation', None)
        # articulation = None
        texture = mesh_dict.get('texture', None)
        mass = mesh_dict.get('mass', 1.0)

        stage = convert_mesh_to_usd(stage, usd_internal_path,
                                    mesh_obj_i.vertices, mesh_obj_i.faces,
                                    collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4),
                                    apply_debug_torque=False, debug_torque_value=30.0, texture=texture,
                                    usd_internal_art_reference_path=f"/World/{mesh_id}",
                                    add_damping=True)


    stage.Save()


    success = UsdUtils.CreateNewUsdzPackage(f"{usd_file_path}",
                                            usd_file_path.replace(".usd", ".usdz"))

    if success:
        print(f"Successfully created USDZ: {usd_file_path.replace('.usd', '.usdz')}")
    else:
        print("Failed to create USDZ.")

def save_door_frame_to_usd(
    usd_file_path,
    mesh_info_dict_door,
    mesh_info_dict_door_frame,
    door_id,
    door_frame_id
):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
    stage = Usd.Stage.CreateNew(usd_file_path)

    world_base_prim = UsdGeom.Xform.Define(stage, "/World")

    # set default prim to World
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World"))

    usd_internal_path_door = f"/World/{door_id}"
    usd_internal_path_door_frame = f"/World/{door_frame_id}"


    mesh_dict_door = mesh_info_dict_door
    mesh_obj_door = mesh_dict_door['mesh']
    articulation_door = mesh_dict_door.get('articulation', None)
    texture_door = mesh_dict_door.get('texture', None)

    mesh_dict_door_frame = mesh_info_dict_door_frame
    mesh_obj_door_frame = mesh_dict_door_frame['mesh']
    texture_door_frame = mesh_dict_door_frame.get('texture', None)

    stage = door_frame_to_usd(
        stage,
        usd_internal_path_door,
        usd_internal_path_door_frame,
        mesh_obj_door,
        mesh_obj_door_frame,
        articulation_door,
        texture_door,
        texture_door_frame
    )

    stage.Save()


    success = UsdUtils.CreateNewUsdzPackage(f"{usd_file_path}",
                                            usd_file_path.replace(".usd", ".usdz"))

    if success:
        print(f"Successfully created USDZ: {usd_file_path.replace('.usd', '.usdz')}")
    else:
        print("Failed to create USDZ.")





def get_room_layout_scene_usd_separate_from_layout(layout_json_path: str, usd_collection_dir: str):
    """
    Create a room layout scene from a dictionary of mesh information.
    """
    with open(layout_json_path, 'r') as f:
        layout_data = json.load(f)

    layout_dir = os.path.dirname(layout_json_path)

    floor_plan = dict_to_floor_plan(layout_data)
    current_layout = floor_plan

    mesh_info_dict = export_layout_to_mesh_dict_list_no_object_transform_v2(current_layout, layout_dir)

    rigid_object_property_dict = {}
    rigid_object_transform_dict = {}

    os.makedirs(usd_collection_dir, exist_ok=True)

    room_base_ids = [mesh_id for mesh_id in mesh_info_dict.keys() if mesh_id.startswith("door_") or mesh_id.startswith("wall_room_") or mesh_id.startswith("window_") or mesh_id.startswith("floor_")]
    rigid_object_ids = [mesh_id for mesh_id in mesh_info_dict.keys() if mesh_id not in room_base_ids]

    door_ids = []
    door_frame_ids = []

    for room_base_id in room_base_ids:
        if room_base_id.startswith("door_"):
            if room_base_id.endswith("_frame"):
                door_frame_ids.append(room_base_id)
            else:
                door_ids.append(room_base_id)
            continue

        usd_file_path = f"{usd_collection_dir}/{room_base_id}.usd"
        save_usd_with_ids(usd_file_path, mesh_info_dict, [room_base_id])

    for rigid_object_id in rigid_object_ids:
        usd_file_path = f"{usd_collection_dir}/{rigid_object_id}.usd"
        rigid_object_property_dict[rigid_object_id] = {
            "static": mesh_info_dict[rigid_object_id]['static'],
            "mass": mesh_info_dict[rigid_object_id]['mass'],
        }
        rigid_object_transform_dict[rigid_object_id] = mesh_info_dict[rigid_object_id]["transform"]
        mesh_info_dict[rigid_object_id]['static'] = False
        save_usd_with_ids(usd_file_path, mesh_info_dict, [rigid_object_id])


    for door_id, door_frame_id in zip(door_ids, door_frame_ids):

        save_door_frame_to_usd(
            usd_file_path=f"{usd_collection_dir}/{door_id}.usd",
            mesh_info_dict_door=mesh_info_dict[door_id],
            mesh_info_dict_door_frame=mesh_info_dict[door_frame_id],
            door_id=door_id,
            door_frame_id=door_frame_id
        )

    with open(os.path.join(usd_collection_dir, "rigid_object_property_dict.json"), "w") as f:
        json.dump(rigid_object_property_dict, f, indent=4)

    with open(os.path.join(usd_collection_dir, "rigid_object_transform_dict.json"), "w") as f:
        json.dump(rigid_object_transform_dict, f, indent=4)

    return {
        "status": "success",
        "message": f"Room layout scene created successfully",
    }


def get_layout_scene_loaded(layout_json_path: str):
    """
    Create a room layout scene from a dictionary of mesh information.
    """
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
    import omni
    with open(layout_json_path, 'r') as f:
        layout_data = json.load(f)

    layout_dir = os.path.dirname(layout_json_path)

    floor_plan = dict_to_floor_plan(layout_data)
    current_layout = floor_plan
    mesh_info_dict = export_layout_to_mesh_dict_list_v2(current_layout, layout_dir)

    stage = Usd.Stage.CreateInMemory()


    world_base_prim = UsdGeom.Xform.Define(stage, "/World")

    # set default prim to World
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    collision_approximation = "sdf"

    track_ids = []
    door_ids = []
    door_frame_ids = []

    print(f"mesh_info_dict: {mesh_info_dict.keys()}")

    for mesh_id in mesh_info_dict:
        if mesh_id.startswith("wall_room_") or mesh_id.startswith("window_") or mesh_id.startswith("floor_"):
            usd_internal_path = f"/World/{mesh_id}"
        elif mesh_id.startswith("door_"):
            if mesh_id.endswith("_frame"):
                door_frame_ids.append(mesh_id)
            else:
                door_ids.append(mesh_id)
            continue
        else:
            track_ids.append(mesh_id)
            usd_internal_path = f"/World/{mesh_id}"
        mesh_dict = mesh_info_dict[mesh_id]
        mesh_obj_i = mesh_dict['mesh']
        static = mesh_dict['static']
        articulation = mesh_dict.get('articulation', None)
        texture = mesh_dict.get('texture', None)
        mass = mesh_dict.get('mass', 1.0)

        print(f"usd_internal_path: {usd_internal_path}")

        stage = convert_mesh_to_usd(stage, usd_internal_path,
                                    mesh_obj_i.vertices, mesh_obj_i.faces,
                                    collision_approximation, static, articulation, mass=mass, physics_iter=(16, 4),
                                    apply_debug_torque=False, debug_torque_value=30.0, texture=texture,
                                    usd_internal_art_reference_path=f"/World/{mesh_id}")

    door_ids = sorted(door_ids)
    door_frame_ids = sorted(door_frame_ids)

    for door_id, door_frame_id in zip(door_ids, door_frame_ids):
        usd_internal_path_door = f"/World/{door_id}"
        usd_internal_path_door_frame = f"/World/{door_frame_id}"


        mesh_dict_door = mesh_info_dict[door_id]
        mesh_obj_door = mesh_dict_door['mesh']
        articulation_door = mesh_dict_door.get('articulation', None)
        texture_door = mesh_dict_door.get('texture', None)

        mesh_dict_door_frame = mesh_info_dict[door_frame_id]
        mesh_obj_door_frame = mesh_dict_door_frame['mesh']
        texture_door_frame = mesh_dict_door_frame.get('texture', None)

        stage = door_frame_to_usd(
            stage,
            usd_internal_path_door,
            usd_internal_path_door_frame,
            mesh_obj_door,
            mesh_obj_door_frame,
            articulation_door,
            texture_door,
            texture_door_frame,
        )

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    omni.usd.get_context().attach_stage_with_callback(stage_id)

    # Set the world axis of the stage root layer to Z
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    return track_ids

def get_all_prim_paths(ids):
    # Get all prim paths in the stage

    prim_paths = [f"/World/{id}" for id in ids]
    return prim_paths

def get_prim(prim_path):
    import omni
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"Prim at path {prim_path} is not valid.")
        return None
    return prim


def quaternion_angle(q1, q2):
    """
    Calculate the angle between two quaternions.

    Parameters:
    q1, q2: Lists or arrays of shape [w, x, y, z] representing quaternions

    Returns:
    angle: The angle in radians between the two quaternions
    """
    # Convert lists to numpy arrays if they aren't already
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate the relative quaternion: q_rel = q2 * q1^(-1)
    q1_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])  # Inverse of a normalized quaternion

    # Quaternion multiplication for q_rel = q2 * q1_inv
    q_rel = np.array([
        q2[0] * q1_inv[0] - q2[1] * q1_inv[1] - q2[2] * q1_inv[2] - q2[3] * q1_inv[3],
        q2[0] * q1_inv[1] + q2[1] * q1_inv[0] + q2[2] * q1_inv[3] - q2[3] * q1_inv[2],
        q2[0] * q1_inv[2] - q2[1] * q1_inv[3] + q2[2] * q1_inv[0] + q2[3] * q1_inv[1],
        q2[0] * q1_inv[3] + q2[1] * q1_inv[2] - q2[2] * q1_inv[1] + q2[3] * q1_inv[0]
    ])

    # The angle can be calculated from the scalar part (real part) of the relative quaternion
    angle = 2 * np.arccos(min(abs(q_rel[0]), 1.0))

    return angle * 180 / np.pi  # Convert to degrees




def get_all_prims_with_paths(ids):
    # implement this function to get all prims in the stage
    prim_paths = get_all_prim_paths(ids)
    prims = []
    for prim_path in prim_paths:
        prim = get_prim(prim_path)
        prims.append(prim)
    return prims, prim_paths

def extract_position_orientation(transform):
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    position = Gf.Vec3d(transform.ExtractTranslation())
    rotation = transform.ExtractRotationQuat()
    orientation = Gf.Quatd(rotation.GetReal(), *rotation.GetImaginary())
    return position, orientation


def start_simulation_and_track(
        prims, prim_paths,
        simulation_steps=2000,
        longterm_equilibrium_steps=20,
        stable_position_limit=0.2, stable_rotation_limit=8.0,
        early_stop_unstable_exemption_prim_paths=[]
    ):

    import omni
    import omni.kit.app
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils
    app = omni.kit.app.get_app()

    # Reset and initialize the simulation
    stage = omni.usd.get_context().get_stage()

    # Get the timeline interface
    timeline = omni.timeline.get_timeline_interface()
    # Stop the timeline if it's currently playing
    if timeline.is_playing():
        timeline.stop()
    # Reset the simulation to initial state
    timeline.set_current_time(0.0)
    # Wait a moment for the reset to complete
    import time
    time.sleep(0.1)
    # Define a list to store the traced data
    traced_data_all = {}
    init_data = {}

    # Start the simulation
    timeline.play()
    # Initialize variables for tracking the previous position for speed calculation
    elapsed_steps = 0
    init = True

    early_stop = False
    while not early_stop and elapsed_steps < simulation_steps:
        # Get the current time code
        current_time_code = Usd.TimeCode.Default()
        # Get current position and orientation
        traced_data_frame_prims = []
        for prim in prims:
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(current_time_code)
            traced_data_frame_prim = extract_position_orientation(transform)
            traced_data_frame_prims.append(traced_data_frame_prim)
        for prim_i, (position, orientation) in enumerate(traced_data_frame_prims):
            # Calculate speed if previous position is available

            prim_path = prim_paths[prim_i]

            traced_data = traced_data_all.get(prim_path, [])


            if init:
                init_data[prim_path] = {}
                init_data[prim_path]["position"] = [position[0], position[1], position[2]]
                init_data[prim_path]["orientation"] = [orientation.GetReal(),
                                                         orientation.GetImaginary()[0],
                                                         orientation.GetImaginary()[1],
                                                         orientation.GetImaginary()[2]
                                                         ]
                relative_position = 0.
                relative_orientation = 0.

                position_cur = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])

                orientation_cur = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])

            else:
                position_cur = np.array([position[0], position[1], position[2]])
                position_init = np.array([init_data[prim_path]["position"][0],
                                          init_data[prim_path]["position"][1],
                                          init_data[prim_path]["position"][2]])

                orientation_cur = np.array([orientation.GetReal(),
                                            orientation.GetImaginary()[0],
                                            orientation.GetImaginary()[1],
                                            orientation.GetImaginary()[2]
                                            ])
                orientation_init = np.array([init_data[prim_path]["orientation"][0],
                                             init_data[prim_path]["orientation"][1],
                                             init_data[prim_path]["orientation"][2],
                                             init_data[prim_path]["orientation"][3]
                                             ])

                position_last = traced_data[0]["position_last"]
                orientation_last = traced_data[0]["orientation_last"]

                relative_position_last = position_cur - position_last
                relative_orientation_last = quaternion_angle(orientation_cur, orientation_last)

                relative_position_last = float(np.linalg.norm(relative_position_last))
                relative_orientation_last = float(relative_orientation_last)

                relative_position = position_cur - position_init
                relative_orientation = quaternion_angle(orientation_cur, orientation_init)

                relative_position = float(np.linalg.norm(relative_position))
                relative_orientation = float(relative_orientation)


            traced_data.append({
                "position": position_cur.copy(),
                "orientation": orientation_cur.copy(),
                "d_position": relative_position,
                "d_orientation": relative_orientation,
                "position_last": position_cur.copy(),
                "orientation_last": orientation_cur.copy(),
            })

            if traced_data[-1]["d_position"] > stable_position_limit or \
               traced_data[-1]["d_orientation"] > stable_rotation_limit:
                traced_data[-1]["stable"] = False
            else:
                traced_data[-1]["stable"] = True

            if not init:
                traced_data[-1]["relative_position_last"] = relative_position_last
                traced_data[-1]["relative_orientation_last"] = relative_orientation_last
                if relative_position_last < 1e-3 and relative_orientation_last < 1e-3:
                    traced_data[-1]["shortterm_equilibrium"] = True
                else:
                    traced_data[-1]["shortterm_equilibrium"] = False


            if len(traced_data) > longterm_equilibrium_steps:
                traced_data.pop(0)

                longterm_equilibrium = True
                for trace_item in traced_data:
                    longterm_equilibrium = longterm_equilibrium and trace_item["shortterm_equilibrium"]

                traced_data[-1]["longterm_equilibrium"] = longterm_equilibrium
            else:
                traced_data[-1]["longterm_equilibrium"] = False
            traced_data_all[prim_path] = traced_data

        all_longterm_equilibrium = True

        for prim_path, traced_data in traced_data_all.items():
            all_longterm_equilibrium = all_longterm_equilibrium and traced_data[-1]["longterm_equilibrium"]

        if all_longterm_equilibrium:
            print("early stop: all longterm equilibrium")
            early_stop = True


        existing_stable = True

        for prim_path, traced_data in traced_data_all.items():
            if prim_path not in early_stop_unstable_exemption_prim_paths and not traced_data[-1]["stable"]:
                print(f"early stop: unstable prim: {prim_path}")
                existing_stable = False
                break

        if not existing_stable:
            early_stop = True

        if init:
            init = False


        # Step the simulation forward by one frame


        # Update the simulation by one frame
        app.update()

        # Also step the timeline forward if needed
        current_time = timeline.get_current_time()
        time_step = 1.0 / 60.0  # Assuming 60 FPS
        timeline.set_current_time(current_time + time_step)


        # Increment the elapsed time
        elapsed_steps += 1

        print(f"\relapsed steps: {elapsed_steps:05d}/{simulation_steps:05d}", end="")

    traced_data_all_final = {}

    for prim_path, traced_data in traced_data_all.items():

        traced_data_all_final[prim_path] = {}
        traced_data_all_final[prim_path]["final_position"] = np.array(traced_data[-1]["position"]).reshape(3)
        traced_data_all_final[prim_path]["final_orientation"] = np.array(traced_data[-1]["orientation"]).reshape(4)
        traced_data_all_final[prim_path]["stable"] = traced_data[-1]["stable"]

        traced_data_all_final[prim_path]["initial_position"] = np.array(init_data[prim_path]["position"]).reshape(3)
        traced_data_all_final[prim_path]["initial_orientation"] = np.array(init_data[prim_path]["orientation"]).reshape(4)

        position_list = [np.array(traced_data[trace_idx]["position"]).reshape(3) for trace_idx in range(len(traced_data))]
        orientation_list = [np.array(traced_data[trace_idx]["orientation"]).reshape(4) for trace_idx in range(len(traced_data))]

        traced_data_all_final[prim_path]["position_traj"] = np.array(position_list).reshape(-1, 3).astype(np.float32)
        traced_data_all_final[prim_path]["orientation_traj"] = np.array(orientation_list).reshape(-1, 4).astype(np.float32)

    # Stop the simulation
    timeline.stop()

    return traced_data_all_final



def generate_physics_statistics(traced_data_all, track_ids):
    """
    Generate physics statistics from traced simulation data.

    Args:
        traced_data_all: Dictionary mapping prim paths to traced data
        track_ids: List of object IDs that were tracked

    Returns:
        Dictionary containing physics statistics in the desired format
    """
    statistics = {
        "objects": {},
        "total_objects": 0,
        "stable_objects": 0,
        "unstable_objects": 0,
        "stability_ratio": 0.0
    }

    # Generate statistics for each object
    for object_id, (prim_path, traced_data) in zip(track_ids, traced_data_all.items()):
        # Extract data
        initial_pos = traced_data["initial_position"]
        final_pos = traced_data["final_position"]
        initial_orient = traced_data["initial_orientation"]
        final_orient = traced_data["final_orientation"]
        stable = traced_data["stable"]

        # Calculate position offset
        position_offset = (final_pos - initial_pos).tolist()
        position_offset_magnitude = float(np.linalg.norm(final_pos - initial_pos))

        # Calculate orientation angle offset using quaternion_angle function
        orientation_angle_offset = float(quaternion_angle(initial_orient, final_orient))


        # Store statistics for this object
        statistics["objects"][object_id] = {
            "stable": bool(stable),
            "position_offset": position_offset,
            "position_offset_magnitude": position_offset_magnitude,
            "orientation_angle_offset": orientation_angle_offset,
        }

        # Update counters
        statistics["total_objects"] += 1
        if stable:
            statistics["stable_objects"] += 1
        else:
            statistics["unstable_objects"] += 1

    # Calculate stability ratio
    if statistics["total_objects"] > 0:
        statistics["stability_ratio"] = statistics["stable_objects"] / statistics["total_objects"]


    return statistics



def simulate_the_scene(track_ids):
    """
    Simulate the scene.
    """
    from pxr import Gf, Usd, UsdGeom, Vt, UsdPhysics, PhysxSchema, UsdUtils, Sdf, UsdShade
    import omni
    stage = omni.usd.get_context().get_stage()

    prims, prim_paths = get_all_prims_with_paths(track_ids)
    traced_data_all = start_simulation_and_track(
        prims, prim_paths, simulation_steps=120, longterm_equilibrium_steps=120,
        early_stop_unstable_exemption_prim_paths=prim_paths
    )

    unstable_prims = []
    unstable_object_ids = []
    for object_id, (prim_path, traced_data) in zip(track_ids, traced_data_all.items()):
        if not traced_data["stable"]:
            unstable_prims.append(os.path.basename(prim_path))
            unstable_object_ids.append(object_id)

    if len(unstable_prims) > 0:
        next_step_message = f"""
The scene is unstable: {unstable_prims};
"""
    else:
        next_step_message = "The scene is stable."

    return {
        "status": "success",
        "message": "Scene simulated successfully!",
        "unstable_objects": unstable_object_ids,
        "next_step": next_step_message,
        "traced_data_all": traced_data_all,
    }