extends CharacterBody3D

@export_subgroup("Properties")
@export var movement_speed = 5
@export var jump_strength = 8
@export var gravity_force = 20.0

@export_subgroup("Weapons")
@export var weapons: Array[Weapon] = []

@export var mouse_sensitivity := 700
@export var gamepad_sensitivity := 0.075
@export var aimbot_enabled := false
@export var aimbot_smoothness := 5.0

var fired_this_frame := false
var weapon: Weapon
var weapon_index := 0
var movement_velocity: Vector3
var rotation_target: Vector3
var input_mouse: Vector2
var gravity: float = 0
var health: int = 100
var previous_fov := 0.0

var jump_single := true
var jump_double := true
var mouse_captured := true
var container_offset = Vector3(1.2, -1.1, -2.75)
var previously_floored := false
var tween: Tween

signal health_updated

@onready var camera = $Head/Camera
@onready var head = $Head
@onready var raycast = $Head/Camera/RayCast
@onready var muzzle = $Head/Camera/SubViewportContainer/SubViewport/CameraItem/Muzzle
@onready var container = $Head/Camera/SubViewportContainer/SubViewport/CameraItem/Container
@onready var sound_footsteps = $SoundFootsteps
@onready var blaster_cooldown = $Cooldown
@export var crosshair: TextureRect

func _ready():
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	weapon = weapons[weapon_index]
	initiate_change_weapon(weapon_index)
	print("Logger autoload status:", Engine.has_singleton("Logger"))

func _physics_process(delta):
	handle_controls(delta)
	handle_gravity(delta)
	if aimbot_enabled:
		aimbot_look(delta)
	else:
		log_normal_aim_data(delta)

	var applied_velocity = velocity.lerp(transform.basis * movement_velocity, delta * 10)
	applied_velocity.y = -gravity
	velocity = applied_velocity
	move_and_slide()

	camera.rotation.z = lerp_angle(camera.rotation.z, -input_mouse.x * 25 * delta, delta * 5)
	camera.rotation.x = lerp_angle(camera.rotation.x, rotation_target.x, delta * 25)
	rotation.y = lerp_angle(rotation.y, rotation_target.y, delta * 25)

	container.position = lerp(container.position, container_offset - (basis.inverse() * applied_velocity / 30), delta * 10)
	camera.position.y = lerp(camera.position.y, 0.0, delta * 5)

	if is_on_floor():
		if gravity > 1 and !previously_floored:
			Audio.play("sounds/land.ogg")
			camera.position.y = -0.1
		sound_footsteps.stream_paused = abs(velocity.x) <= 1 and abs(velocity.z) <= 1
	else:
		sound_footsteps.stream_paused = true

	previously_floored = is_on_floor()

	if position.y < -10:
		get_tree().reload_current_scene()

func _input(event):
	if event is InputEventMouseMotion and mouse_captured:
		input_mouse = event.relative / mouse_sensitivity
		rotation_target.y -= event.relative.x / mouse_sensitivity
		rotation_target.x -= event.relative.y / mouse_sensitivity

	if event.is_action_pressed("mouse_capture"):
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		mouse_captured = true

	if event.is_action_pressed("mouse_capture_exit"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		mouse_captured = false
		input_mouse = Vector2.ZERO

	if event.is_action_pressed("toggle_aimbot"):
		aimbot_enabled = !aimbot_enabled
		print("Aimbot Enabled: ", aimbot_enabled)

func handle_controls(_delta):
	var input := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	movement_velocity = Vector3(input.x, 0, input.y).normalized() * movement_speed

	var rotation_input := Input.get_vector("camera_right", "camera_left", "camera_down", "camera_up")
	rotation_target -= Vector3(-rotation_input.y, -rotation_input.x, 0).limit_length(1.0) * gamepad_sensitivity
	rotation_target.x = clamp(rotation_target.x, deg_to_rad(-90), deg_to_rad(90))

	action_shoot()
	if Input.is_action_just_pressed("jump"):
		if jump_single or jump_double:
			Audio.play("sounds/jump_a.ogg, sounds/jump_b.ogg, sounds/jump_c.ogg")
		if jump_double:
			gravity = -jump_strength
			jump_double = false
		if jump_single:
			action_jump()

	action_weapon_toggle()

func handle_gravity(delta):
	gravity += gravity_force * delta
	if gravity > 0 and is_on_floor():
		jump_single = true
		gravity = 0

func action_jump():
	gravity = -jump_strength
	jump_single = false
	jump_double = true

func action_shoot():
	fired_this_frame = false  # reset at start of frame

	if Input.is_action_pressed("shoot") and blaster_cooldown.is_stopped():
		fired_this_frame = true  # Mark that a shot actually occurred

		Audio.play(weapon.sound_shoot)
		container.position.z += 0.25
		camera.rotation.x += 0.025
		movement_velocity += Vector3(0, 0, weapon.knockback)

		muzzle.play("default")
		muzzle.rotation_degrees.z = randf_range(-45, 45)
		muzzle.scale = Vector3.ONE * randf_range(0.40, 0.75)
		muzzle.position = container.position - weapon.muzzle_position

		blaster_cooldown.start(weapon.cooldown)

		for n in weapon.shot_count:
			raycast.target_position.x = randf_range(-weapon.spread, weapon.spread)
			raycast.target_position.y = randf_range(-weapon.spread, weapon.spread)
			raycast.force_raycast_update()
			if raycast.is_colliding():
				var collider = raycast.get_collider()
				if collider.has_method("damage"):
					collider.damage(weapon.damage)
				var impact = preload("res://objects/impact.tscn").instantiate()
				impact.play("shot")
				get_tree().root.add_child(impact)
				impact.position = raycast.get_collision_point() + raycast.get_collision_normal() / 10
				impact.look_at(camera.global_transform.origin, Vector3.UP, true)


func action_weapon_toggle():
	if Input.is_action_just_pressed("weapon_toggle"):
		weapon_index = wrap(weapon_index + 1, 0, weapons.size())
		initiate_change_weapon(weapon_index)
		Audio.play("sounds/weapon_change.ogg")

func initiate_change_weapon(index):
	weapon_index = index
	tween = get_tree().create_tween().set_ease(Tween.EASE_OUT_IN)
	tween.tween_property(container, "position", container_offset - Vector3(0, 1, 0), 0.1)
	tween.tween_callback(change_weapon)

func change_weapon():
	weapon = weapons[weapon_index]
	for n in container.get_children():
		container.remove_child(n)
	var weapon_model = weapon.model.instantiate()
	container.add_child(weapon_model)
	weapon_model.position = weapon.position
	weapon_model.rotation_degrees = weapon.rotation
	for child in weapon_model.find_children("*", "MeshInstance3D"):
		child.layers = 2
	raycast.target_position = Vector3(0, 0, -1) * weapon.max_distance
	crosshair.texture = weapon.crosshair

func log_normal_aim_data(delta):
	if not is_instance_valid(camera) or not is_instance_valid(head):
		return

	var origin = camera.global_transform.origin
	var cam_forward = -camera.global_transform.basis.z

	for enemy in get_tree().get_nodes_in_group("enemies"):
		if not enemy or not enemy.is_inside_tree():
			continue

		var to_enemy = enemy.global_transform.origin - origin
		var direction = to_enemy.normalized()
		var fov_to_enemy = cam_forward.angle_to(direction)
		var snap_delta = abs(fov_to_enemy - previous_fov)
		previous_fov = fov_to_enemy
		if fov_to_enemy > deg_to_rad(60):
			continue

		Logger.log({
			"player_pos": global_transform.origin,
			"enemy_pos": enemy.global_transform.origin,
			"aim_yaw": rotation_target.y,
			"aim_pitch": rotation_target.x,
			"mouse_dx": input_mouse.x,
			"mouse_dy": input_mouse.y,
			"fov_to_target": fov_to_enemy,
			"snap_delta": snap_delta,
			"time_to_kill": -1.0,
			"is_firing": 1 if Input.is_action_pressed("shoot") else 0,
			"enemy_killed": 1 if enemy.health <= 0 else 0,
			"label": "clean"
		})
		break

func aimbot_look(delta):
	if not is_instance_valid(camera) or not is_instance_valid(head) or get_tree() == null:
		return

	var origin = camera.global_transform.origin
	var cam_forward = -camera.global_transform.basis.z
	var closest_enemy: Node3D = null
	var closest_dist = INF

	for enemy in get_tree().get_nodes_in_group("enemies"):
		if not enemy or not enemy.is_inside_tree():
			continue

		var to_enemy = enemy.global_transform.origin - origin
		var distance = to_enemy.length()
		var angle_to_enemy = cam_forward.angle_to(to_enemy.normalized())
		if distance < closest_dist and angle_to_enemy < deg_to_rad(60):
			closest_enemy = enemy
			closest_dist = distance

	if closest_enemy == null:
		print("Aimbot active: No enemy in FOV")
		return

	var vector_to_enemy = closest_enemy.global_transform.origin - origin
	var direction = vector_to_enemy.normalized()
	var fov_to_enemy = cam_forward.angle_to(direction)
	var snap_delta = abs(fov_to_enemy - previous_fov)
	previous_fov = fov_to_enemy

	var mouse_dx = input_mouse.x
	var mouse_dy = input_mouse.y

	var target_rot = Basis().looking_at(direction).get_euler()
	head.rotation.y = target_rot.y
	camera.rotation.x = clamp(target_rot.x, deg_to_rad(-90), deg_to_rad(90))

	var label = "clean"
	if fov_to_enemy < deg_to_rad(1.5) and mouse_dx == 0 and mouse_dy == 0:
		label = "cheat"
	elif fov_to_enemy < deg_to_rad(5):
		label = "suspect"

	print("Logging aimbot data for enemy: ", closest_enemy.name, " | Label: ", label)

	Logger.log({
		"player_pos": global_transform.origin,
		"enemy_pos": closest_enemy.global_transform.origin,
		"aim_yaw": rotation_target.y,
		"aim_pitch": rotation_target.x,
		"mouse_dx": mouse_dx,
		"mouse_dy": mouse_dy,
		"fov_to_target": fov_to_enemy,
		"snap_delta": snap_delta,
		"time_to_kill": -1.0,
		"is_firing": 1 if fired_this_frame else 0,
		"enemy_killed": 1 if closest_enemy.health <= 0 else 0,
		"label": label
	})
