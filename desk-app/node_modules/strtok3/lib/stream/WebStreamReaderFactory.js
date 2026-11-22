import { WebStreamByobReader } from './WebStreamByobReader.js';
import { WebStreamDefaultReader } from './WebStreamDefaultReader.js';
export function makeWebStreamReader(stream, options) {
    try {
        const reader = stream.getReader({ mode: 'byob' });
        if (reader instanceof ReadableStreamDefaultReader) {
            // Fallback to default reader in case `mode: byob` is ignored
            return new WebStreamDefaultReader(reader, options);
        }
        return new WebStreamByobReader(reader, options);
    }
    catch (error) {
        if (error instanceof TypeError) {
            // Fallback to default reader in case `mode: byob` rejected by a `TypeError`
            return new WebStreamDefaultReader(stream.getReader(), options);
        }
        throw error;
    }
}
