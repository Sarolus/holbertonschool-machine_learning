-- Creates a stored procedure that adds a new correction for a student

DROP PROCEDURE IF EXISTS AddBonus;

DELIMITER $$
CREATE PROCEDURE AddBonus (IN student_id INT, IN project_name VARCHAR(255), IN SCORE INT)
BEGIN
    SELECT id INTO @project_id FROM projects WHERE name = project_name;

    IF @project_id IS NULL THEN
	INSERT INTO projects (name) VALUES (project_name);
	SELECT id INTO @project_id FROM projects WHERE name = project_name;
    END IF;

    INSERT INTO corrections (user_id, project_id, score)
    VALUES (student_id, @project_id, score);
END $$

DELIMITER ;