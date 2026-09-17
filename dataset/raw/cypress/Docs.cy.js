describe('Product', () => {
    it('Menu bar', () => {
      cy.viewport(1475, 750)

      cy.visit("https://xvantage.ai/gettingstarted")

      cy.wait(2000)

      //products section

      cy.get('h1')
      .should('have.text', 'Get Started with Ease.')
      .should('have.css', 'font-weight', '800')
      .should('have.css', 'font-size', '55px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.page-header > p')
      .should('have.css', 'font-weight', '400')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get(':nth-child(1) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'What You Can Expect?')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('ul > :nth-child(4)')
      .should('have.css', 'font-weight', '300')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(189, 189, 189)')

      cy.get('.page-header > .q-btn').click()
      cy.wait(2000)
      //Cancel
     cy.get('.q-dialog__inner>.q-card>.q-card__actions>:nth-child(1)>.q-btn__content').click()

     //API reference
     cy.get('.secondary-nav > :nth-child(2) > .q-btn__content > .block').click()

     cy.get('h1')
      .should('have.text', 'Your Ultimate API Reference.')
      .should('have.css', 'font-weight', '800')
      .should('have.css', 'font-size', '55px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.page-header > p')
      .should('have.css', 'font-weight', '400')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get(':nth-child(1) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Designed for BuildersTest Before You Launch')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')
      
      cy.get(':nth-child(2) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Seamless Extensibility')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.column > :nth-child(1) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Test Before You Launch')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.banner-content > .sub-title')
      .should('have.css', 'font-weight', '600')
      .should('have.css', 'font-size', '40px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')


      //developer 
      cy.get('.secondary-nav > :nth-child(3) > .q-btn__content > .block').click()

      cy.get('h1')
      .should('have.text', 'Master the Art of Development.')
      .should('have.css', 'font-weight', '800')
      .should('have.css', 'font-size', '55px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.page-header > p')
      .should('have.css', 'font-weight', '400')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get(':nth-child(1) > .q-card > .q-card__section--horiz > .text-lg-500')
      .should('have.text', ' Real-World, Actionable Guides ')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')
      
      cy.get(':nth-child(2) > .q-card > .q-card__section--horiz > .text-lg-500')
      .should('have.text', 'Built to Support Your Vision')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.banner-content > .sub-title')
      .should('have.css', 'font-weight', '600')
      .should('have.css', 'font-size', '40px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      //SDK
      cy.get(':nth-child(4) > .q-btn__content > .block').click()

      cy.get('h1')
      .should('have.text', ' SDKs & Libraries for Seamless Integration.')
      .should('have.css', 'font-weight', '800')
      .should('have.css', 'font-size', '55px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.page-header > p')
      .should('have.css', 'font-weight', '400')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.grid-item > .q-card--flat > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Available SDKs & Libraries')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')
      
      cy.get(':nth-child(1) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Built for Cross-Platform Compatibility')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.banner-content > .sub-title')
      .should('have.css', 'font-weight', '600')
      .should('have.css', 'font-size', '40px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      //best practices

      cy.get(':nth-child(5) > .q-btn__content > .block').click()

      cy.get('h1')
      .should('have.text', 'Best Practices for Maximum Efficiency.')
      .should('have.css', 'font-weight', '800')
      .should('have.css', 'font-size', '55px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.page-header > p')
      .should('have.css', 'font-weight', '400')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.grid-item > :nth-child(1) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Why Best Practices Matter')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')
      
      cy.get('.grid-item > .q-card--flat > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Core Principles to Work Smarter')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.banner-content > .sub-title')
      .should('have.css', 'font-weight', '600')
      .should('have.css', 'font-size', '40px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')


      //troubleshooting
      cy.get(':nth-child(6) > .q-btn__content > .block').click()

      cy.get('h1')
      .should('have.text', ' Troubleshoot with Confidence.')
      .should('have.css', 'font-weight', '800')
      .should('have.css', 'font-size', '55px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.page-header > p')
      .should('have.css', 'font-weight', '400')
      .should('have.css', 'font-size', '16px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

      cy.get('.grid-item > .q-card--flat > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Inside the Fix Toolkit')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')
      
      cy.get(':nth-child(1) > :nth-child(1) > .text-lg-500')
      .should('have.text', 'Pro Tips to Stay Ahead')
      .should('have.css', 'font-weight', '500')
      .should('have.css', 'font-size', '22px')
      .should('have.css', 'color', 'rgb(255, 255, 255)')

    })
  
  })