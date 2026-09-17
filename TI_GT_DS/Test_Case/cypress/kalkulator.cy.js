describe('Test Kalkulatora', () => {
  beforeEach(() => {
    cy.visit('http://127.0.0.1:8080'); 
  });

  it('Dodawanie 2 + 3 = 5', () => {
      cy.get('[data-key="2"]').click();
      cy.get('[data-key="+"]').click();
      cy.get('[data-key="3"]').click();
      cy.get('[data-key="="]').click();

      cy.get('.display .output').should('have.text', '5');
  });

  it('Mnożenie 6 × 7 = 42', () => {
      cy.get('[data-key="6"]').click();
      cy.get('[data-key="*"]').click();
      cy.get('[data-key="7"]').click();
      cy.get('[data-key="="]').click();

      cy.get('.display .output').should('have.text', '42');
  });

  it('Czyszczenie ekranu', () => {
      cy.get('[data-key="9"]').click();
      cy.get('[data-key="clear"]').click();

      cy.get('.display .input').should('have.text', '');
      cy.get('.display .output').should('have.text', '');
  });
});