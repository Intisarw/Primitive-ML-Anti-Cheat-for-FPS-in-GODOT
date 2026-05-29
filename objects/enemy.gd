extends Node3D  # or CharacterBody3D if you want physics

@export var player: Node3D
@export var respawn_delay := 2.0  # seconds before the enemy respawns after death

const MAX_HEALTH := 300

@onready var raycast = $RayCast
@onready var muzzle_a = $MuzzleA
@onready var muzzle_b = $MuzzleB

var health := MAX_HEALTH
var time := 0.0
var target_position: Vector3
var spawn_position: Vector3  # where to respawn back to
var destroyed := false


func _ready():
	target_position = position
	spawn_position = position  # remember the original spawn point
	add_to_group("enemies")  # ✅ Group required for aimbot tracking

func _process(delta):
	if destroyed or not is_instance_valid(player):
		return

	# Look at player (aim slightly above feet)
	var player_pos = player.global_transform.origin
	look_at(player_pos + Vector3(0, 0.5, 0), Vector3.UP, true)

	# Floating up/down motion
	target_position.y += cos(time * 5) * 1 * delta
	time += delta
	position = target_position

func damage(amount):
	if destroyed:
		return  # ignore damage while waiting to respawn
	Audio.play("sounds/enemy_hurt.ogg")
	health -= amount
	if health <= 0:
		destroy()

func destroy():
	Audio.play("sounds/enemy_destroy.ogg")
	destroyed = true
	visible = false
	_set_collisions_enabled(false)

	# Wait respawn_delay seconds, then come back
	await get_tree().create_timer(respawn_delay).timeout

	# Safety check: scene may have been torn down during the wait
	if not is_inside_tree():
		return

	respawn()

func respawn():
	health = MAX_HEALTH
	position = spawn_position
	target_position = spawn_position
	visible = true
	_set_collisions_enabled(true)
	destroyed = false

# Toggles every CollisionShape3D under this enemy so a "dead" enemy
# can't be hit by the player's raycast during the respawn delay.
func _set_collisions_enabled(enabled: bool) -> void:
	for child in get_children():
		_toggle_collision_recursive(child, enabled)

func _toggle_collision_recursive(node: Node, enabled: bool) -> void:
	if node is CollisionShape3D:
		node.disabled = not enabled
	for sub in node.get_children():
		_toggle_collision_recursive(sub, enabled)

func _on_timer_timeout():
	if destroyed:
		return  # dead enemy can't shoot back

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
		var _shot_origin = raycast.global_transform.origin
		var _hit_point = raycast.get_collision_point()
