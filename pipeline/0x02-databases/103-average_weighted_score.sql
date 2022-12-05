-- Creates a stored procedure that computes and store the average weighted score for a student

DROP PROCEDURE IF EXISTS ComputeAverageWeightedScoreForUser;

DELIMITER $$
CREATE PROCEDURE ComputeAverageWeightedScoreForUser(IN user_id INTEGER)
BEGIN
	SELECT SUM(score * weight) / SUM(weight) AS average_weighted_score INTO @average_weighted_score
	FROM users
	JOIN corrections ON users.id = corrections.user_id
	JOIN projects ON corrections.project_id = projects.id
	WHERE users.id = user_id;

	UPDATE users SET average_score = @average_weighted_score WHERE id = user_id;
END $$
DELIMITER ;