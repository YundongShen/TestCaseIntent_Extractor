const request = require("supertest");
const baseUrl = "http://localhost:3000";

let heroToken = "";
let villainToken = "";
let eventId = "";

describe("--TESTS--", () => {
	test("Signup Hero", async () => {
		const res = await request(baseUrl).post("/signup").send({
			username: "hero1",
			password: "pass",
			alignment: "hero",
			region: "Downtown",
		});
		expect(res.status).toBe(201);
		expect(res.body.token).toBeDefined();
		heroToken = res.body.token;
	});

	test("Signup Villain", async () => {
		const res = await request(baseUrl).post("/signup").send({
			username: "villain1",
			password: "pass",
			alignment: "villain",
			region: "Downtown",
		});
		expect(res.status).toBe(201);
		expect(res.body.token).toBeDefined();
		villainToken = res.body.token;
	});

	test("Login Hero", async () => {
		const res = await request(baseUrl).post("/login").send({
			username: "hero1",
			password: "pass",
		});
		expect(res.status).toBe(200);
		expect(res.body.token).toBeDefined();
		heroToken = res.body.token; // update token if needed
	});

	test("Login Villain", async () => {
		const res = await request(baseUrl).post("/login").send({
			username: "villain1",
			password: "pass",
		});
		expect(res.status).toBe(200);
		expect(res.body.token).toBeDefined();
		villainToken = res.body.token; // update token if needed
	});

	test("Get Hero Base", async () => {
		const res = await request(baseUrl)
			.get("/base")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(res.body.base).toBeDefined();
	});

	test("Upgrade Hero Base", async () => {
		const res = await request(baseUrl)
			.patch("/upgrade")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ upgrade: "security defenses" });
		expect(res.status).toBe(200);
		expect(res.body.base.level).toBeGreaterThan(1);
	});

	test("Start Hero Mission", async () => {
		const res = await request(baseUrl)
			.post("/mission/start")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ mission: "Stop Bank Robbery" });
		expect(res.status).toBe(200);
		expect(res.body).toHaveProperty("mission", "Stop Bank Robbery");
	});

	test("Hero Battles Villain", async () => {
		const res = await request(baseUrl)
			.post("/battle")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ target: "villain1" });
		expect(res.status).toBe(200);
		expect(["win", "lose"]).toContain(res.body.outcome);
	});

	test("Get Players Nearby (Hero)", async () => {
		const res = await request(baseUrl)
			.get("/players/nearby")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(Array.isArray(res.body.players)).toBe(true);
	});

	test("Recruit Sidekick for Hero", async () => {
		const res = await request(baseUrl)
			.post("/recruit")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ type: "sidekick" });
		expect(res.status).toBe(200);
		expect(res.body.base.sidekicks).toBeGreaterThan(0);
	});

	test("Recruit Minion for Villain", async () => {
		const res = await request(baseUrl)
			.post("/recruit")
			.set("Authorization", `Bearer ${villainToken}`)
			.send({ type: "minion" });
		expect(res.status).toBe(200);
		expect(res.body.base.minions).toBeGreaterThan(0);
	});

	test("Get Map", async () => {
		const res = await request(baseUrl)
			.get("/map")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(res.body.players).toBeDefined();
		expect(res.body.events).toBeDefined();
	});

	test("Attack Territory (Hero)", async () => {
		const res = await request(baseUrl)
			.post("/attack/territory")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(typeof res.body.success).toBe("boolean");
	});

	test("Get Influence (Hero)", async () => {
		const res = await request(baseUrl)
			.get("/influence")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(typeof res.body.influence).toBe("number");
	});

	test("Invest Money (Hero)", async () => {
		const investAmount = 100;
		const res = await request(baseUrl)
			.post("/invest")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ amount: investAmount });
		expect(res.status).toBe(200);

		if (res.body.success) {
			expect(res.body.profit).toBeDefined();
		} else {
			expect(res.body.loss).toBeDefined();
		}
	});

	test("Get Market Items (Hero)", async () => {
		const res = await request(baseUrl)
			.get("/market")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(Array.isArray(res.body.items)).toBe(true);
	});

	test("Buy Market Item (Hero)", async () => {
		const marketRes = await request(baseUrl)
			.get("/market")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(marketRes.status).toBe(200);
		expect(marketRes.body.items.length).toBeGreaterThan(0);
		const item = marketRes.body.items[0];

		const res = await request(baseUrl)
			.post("/buy")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ item: item.name });
		expect(res.status).toBe(200);
		expect(Array.isArray(res.body.inventory)).toBe(true);
	});

	test("Villain Creates Event", async () => {
		const res = await request(baseUrl)
			.post("/create/event")
			.set("Authorization", `Bearer ${villainToken}`)
			.send({ event: "Cyber Attack", region: "Downtown" });
		expect(res.status).toBe(201);
		expect(res.body.event).toBeDefined();
		eventId = res.body.event.id;
	});

	test("Hero Defends Event", async () => {
		if (!eventId) throw new Error("No event created to defend");
		const res = await request(baseUrl)
			.post("/defend")
			.set("Authorization", `Bearer ${heroToken}`)
			.send({ eventId });
		expect(res.status).toBe(200);
		expect(res.body.success).toBeDefined();
	});

	test("Get Leaderboard", async () => {
		const res = await request(baseUrl)
			.get("/leaderboard")
			.set("Authorization", `Bearer ${heroToken}`);
		expect(res.status).toBe(200);
		expect(Array.isArray(res.body.leaderboard)).toBe(true);
	});
});
