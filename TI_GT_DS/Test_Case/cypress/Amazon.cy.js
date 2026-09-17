import "cypress-xpath"

describe("Amazon",()=>{

it("Email Login",()=>{

cy.visit("https://www.amazon.in/")
cy.get("#nav-link-accountList").trigger('mouseover')
cy.get("#nav-link-accountList").click()
cy.get("[type=email").type('grewal.pukhraj@gmail.com')
cy.get(".a-button-input").click()
cy.get("#ap_password").type("Password@1")
cy.get("#signInSubmit").click()

})

it("Product Search, Add to Cart for Checkout",()=>{

cy.visit("https://www.amazon.in/")
cy.get("#twotabsearchtextbox").type("T-shirt")
cy.get("#nav-search-submit-button").click()
cy.xpath("//div[@class='s-widget-container s-spacing-small s-widget-container-height-small celwidget slot=MAIN template=SEARCH_RESULTS widgetId=search-results_1']//a[@class='a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal']").invoke("removeAttr" , "target").click()
cy.get("#add-to-cart-button").click()
cy.xpath("//span[@class='a-button-inner']//input[@value='Proceed to checkout']").click()

})

})