/**
 * Pi extension: a single `web_search` tool backed by a SearXNG JSON-API instance.
 *
 * Loaded explicitly by the Pi harness adapter with `-e <this file>` (and with
 * extension auto-discovery disabled), so it is present ONLY when web search is
 * wanted for a task. The actual search-against-SearXNG logic lives in
 * `web_search_via_searxng_core.ts` (Pi-free, isolation-testable); this file is a
 * thin wrapper that registers the tool and formats results for the model.
 *
 * Deployment-configurable via environment variables:
 *   SEARXNG_BASE_URL                (default http://localhost:8181)
 *   WEB_SEARCH_RESULT_LIMIT         (default 8)
 *   WEB_SEARCH_REQUEST_TIMEOUT_MS   (default 10000)
 *   WEB_SEARCH_SAFESEARCH           (default 1)   0=off 1=moderate 2=strict
 *   WEB_SEARCH_LANGUAGE             (default en)
 *   WEB_SEARCH_CATEGORIES           (default general)
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { performBoundedWebSearch } from "./web_search_via_searxng_core";

const WEB_SEARCH_PARAMETERS = Type.Object({
	query: Type.String({ description: "The search query to send to the web search engine." }),
	result_limit: Type.Optional(
		Type.Number({
			description: "Maximum number of results to return (overrides the deployment default).",
		}),
	),
});

function readEnvironmentInteger(variableName: string, fallbackValue: number): number {
	const rawValue = process.env[variableName];
	if (rawValue === undefined || rawValue.trim() === "") return fallbackValue;
	const parsedValue = Number.parseInt(rawValue, 10);
	return Number.isFinite(parsedValue) ? parsedValue : fallbackValue;
}

function readEnvironmentString(variableName: string, fallbackValue: string): string {
	const rawValue = process.env[variableName];
	return rawValue === undefined || rawValue.trim() === "" ? fallbackValue : rawValue;
}

export default function webSearchViaSearxngExtension(pi: ExtensionAPI) {
	const searxngBaseUrl = readEnvironmentString("SEARXNG_BASE_URL", "http://localhost:8181");
	const defaultResultLimit = readEnvironmentInteger("WEB_SEARCH_RESULT_LIMIT", 8);
	const requestTimeoutMilliseconds = readEnvironmentInteger("WEB_SEARCH_REQUEST_TIMEOUT_MS", 10000);
	const safeSearchLevel = readEnvironmentString("WEB_SEARCH_SAFESEARCH", "1");
	const searchLanguage = readEnvironmentString("WEB_SEARCH_LANGUAGE", "en");
	const searchCategories = readEnvironmentString("WEB_SEARCH_CATEGORIES", "general");

	const registerWebSearchTool = () => {
		pi.registerTool({
			name: "web_search",
			label: "Web Search",
			description:
				"Search the web via a private SearXNG instance and return a bounded list of " +
				"results, each with a title, url, and content snippet.",
			promptSnippet: "Search the web for current information using web_search.",
			promptGuidelines: [
				"Use web_search when the task needs current, external, or citable information.",
				"Prefer a focused query; results are title/url/snippet only, so open URLs with other tools if you need full text.",
			],
			parameters: WEB_SEARCH_PARAMETERS,
			async execute(_toolCallId, params) {
				const effectiveResultLimit =
					typeof params.result_limit === "number" && params.result_limit > 0
						? params.result_limit
						: defaultResultLimit;

				const outcome = await performBoundedWebSearch({
					baseUrl: searxngBaseUrl,
					query: params.query,
					resultLimit: effectiveResultLimit,
					requestTimeoutMilliseconds,
					safeSearchLevel,
					searchLanguage,
					searchCategories,
				});

				if (!outcome.ok) {
					return {
						content: [
							{ type: "text", text: `web_search failed (${outcome.reason}) for query "${params.query}".` },
						],
						details: { query: params.query, failed: true, ...outcome },
					};
				}

				if (outcome.results.length === 0) {
					return {
						content: [{ type: "text", text: `web_search: no results for "${params.query}".` }],
						details: { query: params.query, resultCount: 0 },
					};
				}

				const renderedResults = outcome.results
					.map((result, index) => `${index + 1}. ${result.title}\n   ${result.url}\n   ${result.content}`)
					.join("\n\n");

				return {
					content: [{ type: "text", text: renderedResults }],
					details: { query: params.query, resultCount: outcome.results.length, results: outcome.results },
				};
			},
		});
	};

	pi.on("session_start", () => {
		registerWebSearchTool();
	});
}
