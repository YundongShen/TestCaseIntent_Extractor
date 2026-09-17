describe('Teste de Login no site The Internet', () => {
    beforeEach(() => {
      cy.visit('https://the-internet.herokuapp.com/login');
    });
    
  //Bruna Martins, Keure Passos, Renata Schramm, Taissa Katrine
  
    it('Login com Sucesso', () => {
      cy.get('input[name="username"]').type('tomsmith');
      cy.get('input[name="password"]').type('SuperSecretPassword!');
      cy.get('button[type="submit"]').click();
  
      cy.get('.flash.success').should('contain', 'You logged into a secure area!');
    });
  
    it('Login com Credenciais Incorretas', () => {
      cy.get('input[name="username"]').type('usuario_invalido');
      cy.get('input[name="password"]').type('senha_invalida');
      cy.get('button[type="submit"]').click();
  
      cy.get('.flash.error').should('contain', 'Your username is invalid!');
    });
  
    it('Login com Senha Incorretas', () => {
      cy.get('input[name="username"]').type('tomsmith');
      cy.get('input[name="password"]').type('senha_invalida');
      cy.get('button[type="submit"]').click();
  
      cy.get('.flash.error').should('contain', 'Your password is invalid!');
    });
  
    it('Login com Usuario Incorretas', () => {
      cy.get('input[name="username"]').type('usuario_invalido');
      cy.get('input[name="password"]').type('SuperSecretPassword!');
      cy.get('button[type="submit"]').click();
  
      cy.get('.flash.error').should('contain', 'Your username is invalid!');
    });
  
    it('Validação de Campos Obrigatórios', () => {
      cy.get('button[type="submit"]').click();
      cy.get('.flash.error').should('exist');
    });
  });