package de.entwicklerheld.cartChallenge;

import org.assertj.core.util.Lists;
import org.hamcrest.Matchers;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.MockMvcPrint;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;

import java.util.*;

import static org.hamcrest.Matchers.is;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.junit.Assert.*;

@SpringBootTest
@AutoConfigureMockMvc(print = MockMvcPrint.NONE)
@RunWith(SpringRunner.class)
public class CartServiceTest {


    @Autowired
    private MockMvc mockMvc;

    @Test
    public void testAddToCartScenario1() throws Exception {
        HashMap<String, Object> sessionWithoutCart = new HashMap<>();
        HashMap<String, Object> sessionWithCart = new HashMap<>();
        LinkedList<String> articlesInCart = new LinkedList<>();

        articlesInCart.add("9783770430413");
        articlesInCart.add("4012160162463");
        articlesInCart.add("9006113085102");

        sessionWithCart.put("Cart", articlesInCart);
        sessionWithCart.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");
        sessionWithoutCart.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");

        this.mockMvc.perform(
                post("/update-cart").sessionAttrs(sessionWithCart))
                .andExpect(status().isOk())
                .andReturn();

        this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(articlesInCart)))
                .andReturn();

        HashMap<String, Object> unknownUserSession = new HashMap<>();
        unknownUserSession.put("UserId", "unkown-user-id");
        this.mockMvc.perform(
                get("/get-cart").sessionAttrs(unknownUserSession))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(Lists.emptyList())))
                .andReturn();

        updateAndGet("b6ab966eabbd1dbbf83ae23a402d7d68", false, "1230004026174");

        // Test if first is still there
        this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(articlesInCart)))
                .andReturn();


        updateAndGet("b6ab966eabbd1dbbf83ae23a402d7d68", false, "1230004026174", "6610000256617");
        updateAndGet("b6ab966eabbd1dbbf83ae23a402d7d68", false, "6610000256617");
    }

    @Test
    public void testCheckoutCartScenario2() throws Exception {
        HashMap<String, Object> sessionWithCart = new HashMap<>();
        LinkedList<String> articlesInCart = new LinkedList<>();
        HashMap<String, Object> sessionWithoutCart = new HashMap<>();

        articlesInCart.add("9783770430413");
        articlesInCart.add("4012160162463");
        articlesInCart.add("9006113085102");

        sessionWithCart.put("Cart", articlesInCart);
        sessionWithCart.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");
        sessionWithoutCart.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");

        this.mockMvc.perform(
                post("/update-cart").sessionAttrs(sessionWithCart))
                .andExpect(status().isOk())
                .andReturn();

        this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(articlesInCart)))
                .andReturn();


        this.mockMvc.perform(
                post("/checkout-cart").sessionAttrs(sessionWithCart))
                .andExpect(status().isOk())
                .andReturn();

        CheckoutService checkoutService = CheckoutService.getInstance();
        assertTrue("Expected user id fe5a97dc3be2213bb0889b1e22db087a is known to CheckoutService, but the key was not found.", checkoutService.data.containsKey("fe5a97dc3be2213bb0889b1e22db087a"));
        assertEquals("Expected order " + articlesInCart + " in checkoutService for user fe5a97dc3be2213bb0889b1e22db087a, but was " +checkoutService.data.get("fe5a97dc3be2213bb0889b1e22db087a").toString(), articlesInCart, checkoutService.data.get("fe5a97dc3be2213bb0889b1e22db087a"));

        MvcResult result = this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andReturn();
        String contentAsString = result.getResponse().getContentAsString();
        assertThat("[]", is(contentAsString));


        updateAndGet("b6ab966eabbd1dbbf83ae23a402d7d68", true, "1230004026174", "6610000256617");
        MvcResult result2 = this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andReturn();
        String contentAsString2 = result2.getResponse().getContentAsString();
        assertEquals("[]", contentAsString2);



        updateAndGet("48c07e87a5570bbfc8deea6f36009aab", true, "0045496453152");
        MvcResult result3 = this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andReturn();
        String contentAsString3 = result3.getResponse().getContentAsString();
        assertEquals("[]", contentAsString3);

    }


    @Test
    public void testRecommendationsCartScenario3() throws Exception {
            createScenario3TestData();

        HashMap<String, Object> sessionWithCart1 = new HashMap<>();
        LinkedList<String> articlesInCart1 = new LinkedList<>();
        List<String> expectedRecommendations1 = new LinkedList<>();
        articlesInCart1.add("0000000000001");
        expectedRecommendations1.add("0000000000002");
        sessionWithCart1.put("Cart", articlesInCart1);
        sessionWithCart1.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");
        this.mockMvc.perform(
                get("/get-recommendations").sessionAttrs(sessionWithCart1))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(expectedRecommendations1)))
                .andReturn();



        HashMap<String, Object> sessionWithCart2 = new HashMap<>();
        LinkedList<String> articlesInCart2 = new LinkedList<>();
        List<String> expectedRecommendations2 = new LinkedList<>();
        articlesInCart2.add("0000000000001");
        articlesInCart2.add("0000000000002");
        expectedRecommendations2.add("0000000000007");
        expectedRecommendations2.add("0000000000011");
        sessionWithCart2.put("Cart", articlesInCart2);
        sessionWithCart2.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");
        this.mockMvc.perform(
                get("/get-recommendations").sessionAttrs(sessionWithCart2))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(expectedRecommendations2)))
                .andReturn();



        HashMap<String, Object> sessionWithCart3 = new HashMap<>();
        LinkedList<String> articlesInCart3 = new LinkedList<>();
        List<String> expectedRecommendations3 = new LinkedList<>();
        articlesInCart3.add("0000000000001");
        articlesInCart3.add("0000000000002");
        expectedRecommendations3.add("0000000000007");
        expectedRecommendations3.add("0000000000011");
        sessionWithCart3.put("Cart", articlesInCart3);
        sessionWithCart3.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");
        this.mockMvc.perform(
                get("/get-recommendations").sessionAttrs(sessionWithCart3))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(expectedRecommendations3)))
                .andReturn();



        HashMap<String, Object> sessionWithCart4 = new HashMap<>();
        LinkedList<String> articlesInCart4 = new LinkedList<>();
        List<String> expectedRecommendations4 = new LinkedList<>();
        articlesInCart4.add("0000000000001");
        articlesInCart4.add("0000000000002");
        expectedRecommendations4.add("0000000000007");
        expectedRecommendations4.add("0000000000011");
        sessionWithCart4.put("Cart", articlesInCart4);
        sessionWithCart4.put("UserId", "fe5a97dc3be2213bb0889b1e22db087a");
        this.mockMvc.perform(
                get("/get-recommendations").sessionAttrs(sessionWithCart4))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(expectedRecommendations4)))
                .andReturn();



    }

    private void createScenario3TestData() throws Exception {
        String PRODUCT_01 = "0000000000001";
        String PRODUCT_02 = "0000000000002";
        String PRODUCT_03 = "0000000000003";
        String PRODUCT_04 = "0000000000004";
        String PRODUCT_05 = "0000000000005";
        String PRODUCT_06 = "0000000000006";
        String PRODUCT_07 = "0000000000007";
        String PRODUCT_08 = "0000000000008";
        String PRODUCT_09 = "0000000000009";
        String PRODUCT_10 = "0000000000010";
        String PRODUCT_11 = "0000000000011";
        String PRODUCT_12 = "0000000000012";
        String PRODUCT_13 = "0000000000013";

        updateAndGet("16ab966eabbd1dbbf83ae23a402d7d68", true, PRODUCT_01, PRODUCT_02);
        updateAndGet("26ab966eabbd1dbbf83ae23a402d7d68", true, PRODUCT_02);
        updateAndGet("3b75347199834138a96ab8dff279adea", true, PRODUCT_03, PRODUCT_04);
        updateAndGet("48c07e87a5570bbfc8deea6f36009aab", true, PRODUCT_05);
        updateAndGet("5aad20aa0b071218f67abd70c5fe254b", true, PRODUCT_02);
        updateAndGet("6e86c1d2ff6038da0a458b61315dcce0", true, PRODUCT_03, PRODUCT_04);
        updateAndGet("7185986a060e63c7b9600a2a18364149", true, PRODUCT_06, PRODUCT_07);
        updateAndGet("8fc60058ba2fbb310170e55379104996", true, PRODUCT_08, PRODUCT_09, PRODUCT_10, PRODUCT_11);
        updateAndGet("9b75347199834138a96ab8dff279adea", true, PRODUCT_09, PRODUCT_11);
        updateAndGet("10ab966eabbd1dbbf83ae23a402d7d68", true, PRODUCT_08, PRODUCT_11);
        updateAndGet("1152e7fa035a10f588aa858de142ab71", true, PRODUCT_03, PRODUCT_04);
        updateAndGet("12f08e644cf341a49d6a4ad2f647d920", true, PRODUCT_02);
        updateAndGet("135a97dc3be2213bb0889b1e22db0870", true, PRODUCT_03, PRODUCT_04);
        updateAndGet("145a97dc3be2213bb0889b1e22db087b", true, PRODUCT_08, PRODUCT_09, PRODUCT_10, PRODUCT_11);
        updateAndGet("155a97dc3be2213bb0889b1e22db087c", true, PRODUCT_01, PRODUCT_02);
        updateAndGet("165a97dc3be2213bb0889b1e22db087d", true, PRODUCT_09, PRODUCT_11);
        updateAndGet("170a97dc3be2213bb0889b1e22db087d", true, PRODUCT_12, PRODUCT_13);
        updateAndGet("189a97dc3be2213bb0889b1e22db087d", true, PRODUCT_13);
        updateAndGet("195a97dc3be2213bb0889b1e22db087d", true, PRODUCT_12);
        updateAndGet("205397dc3be2213bb0889b1e22db087d", true, PRODUCT_11);
        updateAndGet("21a927dc3be2213bb0889b1e22db087d", true, PRODUCT_11);
        updateAndGet("22a975dc3be2213bb0889b1e22db087d", true, PRODUCT_11);
        updateAndGet("23a97d6c3be2213bb0889b1e22db087d", true, PRODUCT_11);
        updateAndGet("24a97dc83be2213bb0889b1e22db087d", true, PRODUCT_04, PRODUCT_08);
        updateAndGet("25a97dc39be2213bb0889b1e22db087d", true, PRODUCT_07, PRODUCT_02);
    }

    public void updateAndGet(String userId, boolean doCheckout, String... eans) throws Exception {
        HashMap<String, Object> sessionWithoutCart = new HashMap<>();
        HashMap<String, Object> sessionWithCart = new HashMap<>();

        LinkedList<String> articlesInCart = new LinkedList<>(Arrays.asList(eans));

        sessionWithCart.put("Cart", articlesInCart);
        sessionWithCart.put("UserId", userId);
        sessionWithoutCart.put("UserId", userId);

        this.mockMvc.perform(
                post("/update-cart").sessionAttrs(sessionWithCart))
                .andExpect(status().isOk())
                .andReturn();

        this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(articlesInCart)))
                .andReturn();

        if (!doCheckout){
            return;
        }

        this.mockMvc.perform(
                post("/checkout-cart").sessionAttrs(sessionWithCart))
                .andExpect(status().isOk())
                .andReturn();

        this.mockMvc.perform(
                get("/get-cart").sessionAttrs(sessionWithoutCart))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$", Matchers.equalTo(Lists.emptyList())))
                .andReturn();
    }


    @Rule
    public EntwicklerHeldTestWatcher watcher = new EntwicklerHeldTestWatcher();
}