import type { Readable } from 'node:stream';
import type { OnClose } from '../types.js';
import { AbstractStreamReader } from './AbstractStreamReader.js';
/**
 * Node.js Readable Stream Reader
 * Ref: https://nodejs.org/api/stream.html#readable-streams
 */
export declare class StreamReader extends AbstractStreamReader {
    private s;
    /**
     * Pending requests waiting for data, tracked until completion or cancellation
     */
    private pendingRequests;
    private onEnd;
    private onError;
    private onClose;
    constructor(s: Readable, options?: {
        onClose?: OnClose;
    });
    /**
     * Read chunk from stream
     * @param buffer Target Uint8Array (or Buffer) to store data read from stream in
     * @param mayBeLess - If true, may fill the buffer partially
     * @returns Number of bytes read
     */
    protected readFromStream(buffer: Uint8Array, mayBeLess: boolean): Promise<number>;
    /**
     * Process deferred read request
     * @param request Deferred read request
     */
    private readDeferred;
    private removeRequest;
    private removeLifecycleListeners;
    private reject;
    abort(): Promise<void>;
}
