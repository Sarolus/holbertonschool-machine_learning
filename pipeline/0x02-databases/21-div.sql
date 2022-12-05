-- Creates a function that divides the first number by the second or returns 0 if the second number is 0

DROP FUNCTION IF EXISTS SafeDiv;

DELIMITER $$
CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS DOUBLE

BEGIN
    DECLARE result DOUBLE;
    IF b = 0 THEN
	SET result = 0;
    ELSE
	SET result = a / b;
    END IF;
    RETURN result;
END $$

DELIMITER ;