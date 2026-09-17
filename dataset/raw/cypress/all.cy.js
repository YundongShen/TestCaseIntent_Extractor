describe('Basic Tests', () => {
  before(() => {
    cy.visit('https://demo.x-cart.com/demo/home.php')
  })

  it('should load the home page and verify the title', () => {
    cy.title().should('eq', 'Your Company Name')
    //should verify the home page content
    cy.get('#content-container').should('be.visible')
    cy.url().should('include', 'demo.x-cart.com')
  })
})

describe('Functional Tests', () => {
  before(() => {
    cy.visit('https://demo.x-cart.com/demo/home.php')
  })

  it('should log in a user', () => {
      cy.get('.header-links > #href_Sign_in').click()
      cy.login('demo-customer@x-cart.com','customer')
      cy.get('.name').should('contain','Mr. John Smith')
      cy.get('.line3 > .search > form > .text').type('phone{enter}')
      cy.get('.products').should('contain','Apple')
      cy.get('.first > .item-box > .details > .buttons-cell > .buy-now > form > .buttons-row > .main-button > .button-right > .button-left')
      .click()
      cy.get("#ui-id-1").should('contain','item added to cart')
      cy.get('.continue-shopping').click()
      cy.get(':nth-child(6) > .item-box > .details > .buttons-cell > .buy-now > form > .buttons-row > .main-button > .button-right > .button-left > .fa')
      .click()
      cy.get("#ui-id-1").should('contain','item added to cart')
      cy.get('.continue-shopping').click()
      cy.get('.icon').click()
      cy.get('.minicart-buttons > .buttons-row > .main-button').click()
      cy.get('h1').should('contain','Checkout')
      cy.get('#accept_terms').check()
      cy.get('.button-right').click()
      cy.get('#center-main > :nth-child(2)').should('contain','Congratulations! Your order has been successfully placed.')
  })

})

describe('Basic Tests', () => {
  before(() => {
    cy.visit('https://demo.x-cart.com/demo/home.php')
  })

  it('X-cart', () => {

      cy.get('.header-links > [href="register.php"]').click()
      cy.get('#firstname').type("{backspace}")
      cy.get('#lastname').type("{backspace}")
      cy.get('#email').type('g@gmail.com')
      cy.get('#passwd1').type('a')
      cy.get('#passwd2').type('a')
      cy.get('#accept_terms_register').check()
      cy.get('.button-right').click()
      cy.get('#ui-id-1').should('contain',"The required field 'First name' is empty!")
      cy.get('.ui-dialog-buttonset > .ui-button').click()

      cy.get('#firstname').type("gg")
      cy.get('#lastname').type("gg")
      cy.get('#passwd1').type('a')
      cy.get('#passwd2').type('a')
      cy.get('#accept_terms_register').check()
      cy.get('.button-right').click()
      cy.get('#ui-id-1').should('contain','Email address is invalid! Please correct')
      cy.get('.ui-dialog-buttonset > .ui-button').click()
      
      cy.get('.line3 > .search > form > .text').type('vadapaav{enter}')
      cy.get('#top-message > .box').should('contain','Warning: Sorry, your search returned no results. Try again with different criteria.')
  })
})

describe('UI and Visual Tests', () => {
  before(() => {
    cy.visit('https://demo.x-cart.com/demo/home.php')
  })

  it('should ensure header is visible', () => {
      cy.get('#header').should('be.visible')
      cy.get('#footer').should('be.visible')
      cy.get('.image').should('be.visible')
  })
})

describe('UI and Visual Tests', () => {
  before(() => {
    cy.visit('https://demo.x-cart.com/demo/home.php')
  })

  it('should ensure header is visible', () => {
      cy.get('.header-links > #href_Sign_in').click()
      cy.login('demo-customer@x-cart.com','customer')
      cy.get('.name').should('contain','Mr. John Smith')
  })
})
