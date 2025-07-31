<?php
session_start();
include '../Includes/db_connection.php';
if (!isset($_SESSION["admin"])) {
    header("Location: admin_login.php");
    exit();
}

// Fetch doctors and patients
$doctors = mysqli_query($conn, "SELECT * FROM doctors");
$patients = mysqli_query($conn, "SELECT * FROM patients");
?>

<!DOCTYPE html>
<html>
<head>
    <title>Manage Users</title>
    <link rel="stylesheet" href="../assets/css/style.css">
</head>
<body>
    <h2>Manage Users</h2>
    <a href="admin_dashboard.php">Back to Dashboard</a><br><br>

    <h3>Doctors</h3>
    <table border="1">
        <tr><th>ID</th><th>Name</th><th>Email</th></tr>
        <?php while($row = mysqli_fetch_assoc($doctors)) {
            echo "<tr><td>{$row['id']}</td><td>{$row['name']}</td><td>{$row['email']}</td></tr>";
        } ?>
    </table>

    <h3>Patients</h3>
    <table border="1">
        <tr><th>ID</th><th>Name</th><th>Email</th></tr>
        <?php while($row = mysqli_fetch_assoc($patients)) {
            echo "<tr><td>{$row['id']}</td><td>{$row['name']}</td><td>{$row['email']}</td></tr>";
        } ?>
    </table>
</body>
</html>
