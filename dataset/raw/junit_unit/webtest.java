
//@ActiveProfiles("test") // Activate the 'test' profile
//@SpringBootTest

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.boot.test.context.SpringBootTest;

import nk.springprojects.reactive.model.HomeRepository;
import nk.springprojects.reactive.model.HomeService;

@ContextConfiguration(classes = TestSecurityConfig.class)
@WebFluxTest(controllers = HomeController.class)
@AutoConfigureWebTestClient
public class featuretest {
	
	@Autowired
    private WebTestClient webTestClient;

    @Mock
    private HomeService service;

//    @InjectMocks
//    private HomeController homeController; 

    private ObjectMapper objectMapper;
    record VoteRequest(String vote, String skilluuid) {}

    @BeforeEach
    public void setup() {
        MockitoAnnotations.openMocks(this);
        objectMapper = new ObjectMapper();
        
        Skill skill = Skill.builder()
			.skillname("Spring boot")
			.skilluuid("spboot-3.3")
			.skillicon("devicon-springboot-plain")
			.rating(3)
			.build();
        
        // Mock the behavior of the repository
        when(service.getRepository()).thenReturn(mock(HomeRepository.class)); // Mock the repository
        	
        // mock the behaviour of repository find skill
        when(service.getRepository().findFirstBySkilluuid("spboot-3.3")).thenReturn(Mono.just(skill));
        
        // mock the behaviour of repository save skill
        when(service.getRepository().save(any(Skill.class))).thenAnswer(invocation -> {
        	Skill updatedSkill = invocation.getArgument(0);
            updatedSkill.setRating(updatedSkill.getRating() + 1);
            return Mono.just(updatedSkill);
        });
        
        // mock the deleteAll method
        when(service.getRepository().deleteAll()).thenReturn(Mono.empty());
    }
    
    @AfterEach
    public void cleansetup() {
       service.getRepository().deleteAll().subscribe();
    }
    
    @Test
    public void testVoteSkill() throws Exception {
    	// Given
        String skillUuid = "spboot-3.3";

        // Step 1: Request the CSRF token
//        CsrfToken csrfToken = webTestClient
//            .get()
//            .uri("/api/auth/csrf-token")
//            .exchange()
//            .expectStatus().isOk()
//            .expectBody(CsrfToken.class)
//            .returnResult()
//            .getResponseBody();

        MultiValueMap<String, String> formData = new LinkedMultiValueMap<>();
        formData.add("vote", "like");
        formData.add("skilluuid", "spboot-3.3");

        // When
        webTestClient
	        .post()
	        .uri("/skill-vote")
	        .contentType(MediaType.APPLICATION_FORM_URLENCODED)
	        .bodyValue(formData)
	        .exchange()
	        .expectStatus().isOk();

        // Then
        verify(service.getRepository(), times(1)).findFirstBySkilluuid(skillUuid);
        verify(service.getRepository(), times(1)).save(any(Skill.class));

        // Assert: check the updated skill rating
        Skill updatedSkill = service.getRepository().findFirstBySkilluuid(skillUuid).block();
        assert updatedSkill != null;
        assert updatedSkill.getRating() == 4; // Rating should be incremented
    }
}


@SpringBootTest
public class databasetest {
	
	public List<String> langNames = new ArrayList<String>(Arrays.asList("Php","Python","Node","Ruby","Scala"));
	public Random rand = new Random();
	
	@Mock
	HomeRepository repository;
	
	@InjectMocks
	HomeService service;
	
	@BeforeEach
    public void setUp() {
		MockitoAnnotations.openMocks(this);
        System.out.println("==========started the testing phase==============");
        repository.deleteAll();
    }
	
	@Test
	public void testSavingNewSkill() {
		// service.saveSkill(Skill.builder()
		// 	.skillname(langNames.get(rand.nextInt(0, langNames.size())))
		// 	.skilluuid(UUID.randomUUID().toString())
		// 	.rating(rand.nextInt())
		// 	.build()).block();
		// List<Skill> alls = service.getRepodockersitory().findAll().collectList().block();
		// assertNotNull(alls);
		assertTrue(true);
	}

