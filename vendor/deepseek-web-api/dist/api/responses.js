import { errorMessage, errorStatus } from "../utils/errors.js";
import { writeJson, writeSse, writeSseHeaders } from "../utils/http.js";
import { completeCodeproxyResponses, streamCodeproxyResponses } from "./codeproxyResponses.js";
export function responsesErrorEvent(error) {
    return {
        type: "error",
        code: "deepseek_web_error",
        message: errorMessage(error),
        param: null,
    };
}
/** Stream when requested; otherwise return the fully accumulated response object. */
export async function handleResponses(response, body, client) {
    if (body.stream === true) {
        writeSseHeaders(response);
        try {
            for await (const event of streamCodeproxyResponses(body, client)) {
                writeSse(response, event.type, event);
            }
            response.write("data: [DONE]\n\n");
            response.end();
        }
        catch (error) {
            writeSse(response, "error", responsesErrorEvent(error));
            response.end();
        }
        return;
    }
    try {
        const result = await completeCodeproxyResponses(body, client);
        writeJson(response, 200, result);
    }
    catch (error) {
        writeJson(response, errorStatus(error), { error: { message: errorMessage(error) } });
    }
}
//# sourceMappingURL=responses.js.map
