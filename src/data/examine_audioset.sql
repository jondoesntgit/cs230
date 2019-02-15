SELECT COUNT(*) FROM labels_videos;

-- 1,568,047 label -> video mappings
--    - bal_train: 
-- 791,203 videos

-- Get most common labels
SELECT labels.display_name, labels.id, COUNT(labels_videos.id) 
FROM labels_videos INNER JOIN labels ON labels.id = labels_videos.label_id 
WHERE labels_videos.split_id = 2 -- unbalanced train
GROUP BY label_id 
ORDER BY COUNT(labels_videos.id) DESC
LIMIT 30;

-- Get videos with the most labels
SELECT
	videos.video_id as video_slug, 
	COUNT(labels_videos.label_id) as labels
FROM labels_videos 
INNER JOIN videos ON videos.video_id = labels_videos.video_id
GROUP BY video_slug
ORDER BY labels DESC
LIMIT 30
; 

-- See labels for a video
SELECT
	labels.display_name, labels_videos.label_id, labels_videos.start_time_seconds, labels_videos.end_time_seconds
FROM labels
INNER JOIN labels_videos ON labels.id = labels_videos.label_id
WHERE labels_videos.video_id='14AOXhSon20';

-- How many videos are not 'musical'
SELECT
videos.video_id as id
FROM videos
WHERE video_id NOT IN (
	SELECT video_id as id FROM labels_videos
	WHERE label_id IN (137)
)
LIMIT 30;

-- Non-musical videos with a lot of labels
-- (still have a lot of percussion, they're just not labeled as music)
SELECT
	videos.video_id as video_slug, 
	COUNT(labels_videos.label_id) as labels
FROM labels_videos 
INNER JOIN videos ON videos.video_id = labels_videos.video_id
WHERE video_slug NOT IN (
	SELECT video_id as id FROM labels_videos
	WHERE label_id IN (137)
)
GROUP BY video_slug
ORDER BY labels DESC
LIMIT 30
; 

-- Show how many videos have # labels
SELECT labels, COUNT(video_slug) FROM
(
	SELECT
		videos.video_id as video_slug, 
		COUNT(labels_videos.label_id) as labels
	FROM labels_videos 
	INNER JOIN videos ON videos.video_id = labels_videos.video_id
	GROUP BY video_slug
	-- ORDER BY labels DESC -- not necessary
)
GROUP BY labels
ORDER BY labels DESC
; 

-- Only select videos with a certain number of labels
SELECT video_slug, labels FROM 
(
	SELECT
		video_id as video_slug,
		COUNT(label_id) as labels
	FROM labels_videos
	GROUP BY video_slug
)
WHERE labels = 4 -- >, <, =>, <= work too
ORDER BY labels DESC
LIMIT 30
; 

-- Show how many videos are of a certain duration
SELECT 
	ROUND(end_time_seconds - start_time_seconds, 2) as duration,
	COUNT(*)
FROM labels_videos
GROUP BY duration
ORDER BY duration DESC
;
