<?php
session_start();
if (!isset($_SESSION["admin"])) {
    header("Location: admin_login.php");
    exit();
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Admin Dashboard</title>
    <link rel="stylesheet" href="../assets/css/style.css">
</head>
<body>
    <h2>Welcome, <?php echo $_SESSION["admin"]; ?> (Admin)</h2>
    <nav>
        <a href="manage_users.php">Manage Users</a> |
        <a href="../Includes/logout.php">Logout</a>
    </nav>
    <p>This is the Admin Dashboard for Hospital Management System.</p>
</body>
</html>
