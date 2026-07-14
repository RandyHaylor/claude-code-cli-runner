/**
 * Isolation test for the web_search core — NO Pi, NO model in the loop.
 * Exercises `performBoundedWebSearch` directly against a live SearXNG instance
 * (default http://localhost:8181, override with SEARXNG_BASE_URL) and asserts:
 *   - a real query returns bounded {title,url,content} results
 *   - the result_limit bound is honored
 *   - a bad base URL fails cleanly (ok:false) rather than throwing
 *
 * Run on the VM with tsx:  tsx web_search_via_searxng_core.isolation-test.ts
 * Exit code 0 = pass, 1 = fail.
 */

import { performBoundedWebSearch } from "./web_search_via_searxng_core";

const searxngBaseUrl = process.env.SEARXNG_BASE_URL ?? "http://localhost:8181";
let failureCount = 0;

function check(description: string, condition: boolean, detail?: unknown): void {
	if (condition) {
		console.log(`PASS: ${description}`);
	} else {
		failureCount += 1;
		console.error(`FAIL: ${description}`, detail !== undefined ? JSON.stringify(detail) : "");
	}
}

async function main(): Promise<void> {
	const requestedLimit = 3;
	const outcome = await performBoundedWebSearch({
		baseUrl: searxngBaseUrl,
		query: "wikipedia",
		resultLimit: requestedLimit,
		requestTimeoutMilliseconds: 10000,
		safeSearchLevel: "1",
		searchLanguage: "en",
		searchCategories: "general",
	});

	check("live query succeeds (ok:true)", outcome.ok === true, outcome);
	if (outcome.ok) {
		check("returns at least one result", outcome.results.length >= 1, outcome.results.length);
		check(
			`honors result_limit (<= ${requestedLimit})`,
			outcome.results.length <= requestedLimit,
			outcome.results.length,
		);
		const first = outcome.results[0];
		check(
			"first result has title, url, content fields",
			typeof first?.title === "string" && typeof first?.url === "string" && typeof first?.content === "string",
			first,
		);
	}

	const badOutcome = await performBoundedWebSearch({
		baseUrl: "http://127.0.0.1:59999",
		query: "anything",
		resultLimit: 3,
		requestTimeoutMilliseconds: 2000,
		safeSearchLevel: "1",
		searchLanguage: "en",
		searchCategories: "general",
	});
	check("unreachable base URL fails cleanly (ok:false, no throw)", badOutcome.ok === false, badOutcome);

	if (failureCount > 0) {
		console.error(`\n${failureCount} check(s) failed.`);
		process.exit(1);
	}
	console.log("\nAll isolation checks passed.");
}

main().catch((error) => {
	console.error("isolation test threw:", error);
	process.exit(1);
});
