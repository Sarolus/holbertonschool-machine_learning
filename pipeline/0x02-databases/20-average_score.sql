-- Creates a stored procedure that computes and stores average score for a student

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser (IN student_id INT)
BEGIN
    SELECT AVG(score) INTO @average_score FROM corrections WHERE user_id = student_id;
    UPDATE users SET average_score = @average_score WHERE id = student_id;
END $$