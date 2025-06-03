extends Node

var file: FileAccess  # Required in Godot 4

func _ready():
	var timestamp = Time.get_datetime_string_from_system().replace(":", "_").replace(" ", "_")
	var filename = "user://aimbot_log_%s.csv" % timestamp
	file = FileAccess.open(filename, FileAccess.WRITE)
	
	if file:
		file.store_line("timestamp,player_pos,enemy_pos,aim_yaw,aim_pitch,mouse_dx,mouse_dy,fov_to_target,snap_delta,time_to_kill,is_firing,enemy_killed,label")

func log(data: Dictionary):
	if file == null:
		print("Logger: File not ready")
		return

	var line = [
		str(Time.get_ticks_msec() / 1000.0),
		str(data["player_pos"]),
		str(data["enemy_pos"]),
		str(rad_to_deg(data["aim_yaw"])),
		str(rad_to_deg(data["aim_pitch"])),
		str(data["mouse_dx"]),
		str(data["mouse_dy"]),
		str(rad_to_deg(data["fov_to_target"])),
		str(rad_to_deg(data["snap_delta"])),  # ✅ Dictionary access
		str(data["time_to_kill"]),
		str(data["is_firing"]),
		str(data["enemy_killed"]),
		str(data["label"])
	]

	file.store_line(",".join(line))
	file.flush()

	print("Logged line: ", ",".join(line))
