Feature: Start wood processing machine
  Scenario: User starts the machine
    Given the wood processing machine is turned off
    When the user presses the power button
    Then the machine should turn on
    And the indicator light should be green

Feature: Set cutting thickness
  Scenario: User sets the cutting thickness to 2 inches
    Given the wood processing machine is on
    And the machine display shows the default thickness
    When the user sets the cutting thickness to 2 inches
    Then the machine should display "2 inches"
    And the blade should adjust to 2 inches


Feature: Load wood for processing
  Scenario: User loads wood onto the machine
    Given the machine is on and ready
    And the user has selected the processing mode
    When the user loads a wood piece onto the machine feeder
    Then the machine should detect the wood
    And display "Wood detected" on the screen


Feature: Process wood plank
  Scenario: User processes a wood plank
    Given the wood is loaded onto the machine
    And the desired thickness is set
    When the user presses the "Start Processing" button
    Then the machine should process the wood
    And the wood should come out at the specified thickness
