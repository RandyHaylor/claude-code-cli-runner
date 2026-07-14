/**
 * Pure web-search-against-SearXNG logic, with NO dependency on Pi or typebox, so
 * it can be exercised in isolation (no model, no harness) as well as from the Pi
 * `web_search` tool wrapper. Uses only global `fetch` (Node >= 18).
 *
 * `performBoundedWebSearch` returns a discriminated result: either
 * {ok:true, results:[{title,url,content}]} or {ok:false, reason, httpStatus?, timedOut?}.
 */

export interface BoundedWebSearchResult {
	title: string;
	url: string;
	content: string;
}

export interface WebSearchRequestConfiguration {
	baseUrl: string;
	query: string;
	resultLimit: number;
	requestTimeoutMilliseconds: number;
	safeSearchLevel: string;
	searchLanguage: string;
	searchCategories: string;
}

export type WebSearchOutcome =
	| { ok: true; results: BoundedWebSearchResult[] }
	| { ok: false; reason: string; httpStatus?: number; timedOut?: boolean };

interface SearxngRawResult {
	title?: string;
	url?: string;
	content?: string;
}

export async function performBoundedWebSearch(
	configuration: WebSearchRequestConfiguration,
): Promise<WebSearchOutcome> {
	const searchRequestUrl = new URL("/search", configuration.baseUrl);
	searchRequestUrl.searchParams.set("q", configuration.query);
	searchRequestUrl.searchParams.set("format", "json");
	searchRequestUrl.searchParams.set("safesearch", configuration.safeSearchLevel);
	searchRequestUrl.searchParams.set("language", configuration.searchLanguage);
	searchRequestUrl.searchParams.set("categories", configuration.searchCategories);

	const abortController = new AbortController();
	const timeoutHandle = setTimeout(
		() => abortController.abort(),
		configuration.requestTimeoutMilliseconds,
	);

	try {
		const searchResponse = await fetch(searchRequestUrl.toString(), {
			signal: abortController.signal,
			headers: { Accept: "application/json" },
		});

		if (!searchResponse.ok) {
			return { ok: false, reason: `HTTP ${searchResponse.status}`, httpStatus: searchResponse.status };
		}

		const searchPayload = (await searchResponse.json()) as { results?: SearxngRawResult[] };
		const rawResults = Array.isArray(searchPayload.results) ? searchPayload.results : [];
		const results = rawResults.slice(0, configuration.resultLimit).map((result) => ({
			title: result.title ?? "",
			url: result.url ?? "",
			content: result.content ?? "",
		}));
		return { ok: true, results };
	} catch (error) {
		const timedOut = error instanceof Error && error.name === "AbortError";
		const reason = timedOut
			? `timed out after ${configuration.requestTimeoutMilliseconds}ms`
			: error instanceof Error
				? error.message
				: String(error);
		return { ok: false, reason, timedOut };
	} finally {
		clearTimeout(timeoutHandle);
	}
}
