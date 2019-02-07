CREATE TABLE IF NOT EXISTS labels (
	id INTEGER PRIMARY_KEY,
	mid VARCHAR(20) NOT NULL,
	display_name VARCHAR(50) NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	video_id VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS splits (
	id INTEGER PRIMARY_KEY,
	name VARCHAR(50) NOT NULL
);

INSERT INTO splits (name) VALUES 
	('bal_train'),
	('unbal_train'),
	('eval')
	;

CREATE TABLE IF NOT EXISTS labels_videos (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	video_id VARCHAR(20),
	label_id INTEGER,
	split_id INTEGER,
	start_time_seconds REAL,
	end_time_seconds REAL,
	CONSTRAINT fk_videos
		FOREIGN KEY (video_id)
		REFERENCES videos(video_id),
	CONSTRAINT fk_labels
		FOREIGN KEY (label_id)
		REFERENCES labels(id),
	CONSTRAINT fk_splits
		FOREIGN KEY (split_id)
		REFERENCES splits(id)
);
