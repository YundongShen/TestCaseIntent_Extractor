
describe('Frames',()=>{
it('iframe 1',()=>{

cy.visit('https://the-internet.herokuapp.com/iframe')

const iframe = cy.get('#mce_0_ifr')      //  locate the text field and store it in the variable
.its('0.contentDocument.body')           // to locate the iframe body from the 1 document
.should('be.visible')
.then(cy.wrap) ;                        // allows additional actionas such as type and click in the iframes body

iframe.type("Hello {cmd+a}");        // 'cmd+a' = a keystroke which selects the typed text to apply below command
cy.get("[aria-label='Bold']").click();

})

it('iFrame 2 using custom command',()=>{
    cy.visit('https://the-internet.herokuapp.com/iframe');

    cy.getIframe('#mce_0_ifr').type("Hello {cmd+a}");       
    cy.get("[aria-label='Bold']").click();
    
})

it('iframe 3 using Iframe plugin',()=>{
    cy.visit('https://the-internet.herokuapp.com/iframe')

    cy.frameLoaded('#mce_0_ifr');           // to load the frame
    cy.iframe('#mce_0_ifr').type("Hello {cmd+a}");
    cy.get("[aria-label='Bold']").click()

})


})
