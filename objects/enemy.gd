extends Node3D  # or CharacterBody3D if you want physics

@export var player: Node3D

@onready var raycast = $RayCast
@onready var muzzle_a = $MuzzleA
@onready var muzzle_b = $MuzzleB

var health := 300
var time := 0.0
var target_position: Vector3
var destroyed := false


func _ready():
	target_position = position
	add_to_group("enemies")  # ✅ Group required for aimbot tracking

func _process(delta):
	if not is_instance_valid(player):
		return

	# Look at player (aim slightly above feet)
	var player_pos = player.global_transform.origin
	look_at(player_pos + Vector3(0, 0.5, 0), Vector3.UP, true)

	# Floating up/down motion
	target_position.y += cos(time * 5) * 1 * delta
	time += delta
	position = target_position

func damage(amount):
	Audio.play("sounds/enemy_hurt.ogg")
	health -= amount
	if health <= 0 and not destroyed:
		destroy()

func destroy():
	Audio.play("sounds/enemy_destroy.ogg")
	destroyed = true
	queue_free()

func _on_timer_timeout():
	raycast.force_raycast_update()

	if not raycast.is_colliding():
		return

	var collider = raycast.get_collider()
	if not collider or not collider.has_method("damage"):
		return

	# Muzzle flash visuals
	muzzle_a.frame = 0
	muzzle_a.play("default")
	muzzle_a.rotation_degrees.z = randf_range(-45, 45)

	muzzle_b.frame = 0
	muzzle_b.play("default")
	muzzle_b.rotation_degrees.z = randf_range(-45, 45)

	Audio.play("sounds/enemy_attack.ogg")
	collider.damage(5)

	# ML Shot Logging (optional)
	if Engine.has_singleton("GlobalMLData"):
		var shot_origin = raycast.global_transform.origin
		var hit_point = raycast.get_collision_point()
