-- Creates a function that divides the first number by the second or returns 0 if the second number is 0


DELIMITER //

CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
        IF b = 0 THEN
           RETURN 0;
        ELSE
                RETURN (a / b);
        END IF;
END //
DELIMITER ;