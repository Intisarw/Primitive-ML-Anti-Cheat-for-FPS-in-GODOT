extends Node

# Where to save CSV logs. Change this path if you reorganize the project.
const OUTPUT_DIR = "res://ml/data/raw"

var file: FileAccess  # Required in Godot 4

func _ready():
	var timestamp = Time.get_datetime_string_from_system().replace(":", "_").replace(" ", "_")

	# Make sure the output folder exists (no-op if already there)
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(OUTPUT_DIR))

	var filename = "%s/aimbot_log_%s.csv" % [OUTPUT_DIR, timestamp]
	file = FileAccess.open(filename, FileAccess.WRITE)

	if file:
		print("MLLogger: writing CSV to ", ProjectSettings.globalize_path(filename))
		file.store_line("timestamp,player_x,player_y,player_z,enemy_x,enemy_y,enemy_z,aim_yaw,aim_pitch,mouse_dx,mouse_dy,fov_to_target,snap_delta,time_to_kill,is_firing,enemy_killed,label")
	else:
		print("MLLogger: FAILED to open ", filename, " (error ", FileAccess.get_open_error(), ")")

func log_event(data: Dictionary):
	if file == null:
		print("MLLogger: File not ready")
		return

	# Split Vector3 fields into x/y/z so we don't write commas inside CSV fields
	var ppos: Vector3 = data["player_pos"]
	var epos: Vector3 = data["enemy_pos"]

	var line = [
		str(Time.get_ticks_msec() / 1000.0),
		str(ppos.x), str(ppos.y), str(ppos.z),
		str(epos.x), str(epos.y), str(epos.z),
		str(rad_to_deg(data["aim_yaw"])),
		str(rad_to_deg(data["aim_pitch"])),
		str(data["mouse_dx"]),
		str(data["mouse_dy"]),
		str(rad_to_deg(data["fov_to_target"])),
		str(rad_to_deg(data["snap_delta"])),
		str(data["time_to_kill"]),
		str(data["is_firing"]),
		str(data["enemy_killed"]),
		str(data["label"])
	]

	file.store_line(",".join(line))
	file.flush()
