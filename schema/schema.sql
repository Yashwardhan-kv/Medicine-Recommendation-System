
CREATE TABLE Patient (
    id INT AUTO_INCREMENT PRIMARY KEY,
    gender VARCHAR(50),
    symptoms VARCHAR(255),
    medical_history TEXT,
    allergies TEXT,
    recommended_medication VARCHAR(255)
);
