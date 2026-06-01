Feature: Login
  Scenario: Logging in with correct credentials
    Given I am on "/login"
    When I fill in "Username" with "admin"
    And I fill in "Password" with "admin123"
    And I press "Login"
    Then I should be redirected to "/profile"
    And I should see "Welcome, admin"


Scenario: Logging in with incorrect credentials
    Given I am on "/login"
    When I fill in "Username" with "admin"
    And I fill in "Password" with "wrongpassword"
    And I press "Login"
    Then I should see "Invalid credentials"

Scenario: Visiting the profile page without being logged in
    Given I am not logged in
    When I am on "/profile"
    Then I should see "Please log in to access your profile."