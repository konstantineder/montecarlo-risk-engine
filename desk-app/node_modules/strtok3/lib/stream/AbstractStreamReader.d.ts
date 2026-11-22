import type { OnClose } from '../types.js';
export interface IStreamReader {
    /**
     * Peek ahead from the stream. Subsequent reads or peeks can return the retained data.
     * The reader borrows the returned prefix of the target: do not mutate it while retained.
     * @param uint8Array - Uint8Array (or Buffer) to store data read from stream in
     * @param mayBeLess - Allow short reads at EOF; may still wait and combine source chunks
     * @returns Number of bytes peeked; only this prefix of the target contains result bytes.
     * Unlike tokenizer methods, a reader may return retained partial data at known EOF even without mayBeLess.
     */
    peek(uint8Array: Uint8Array, mayBeLess?: boolean): Promise<number>;
    /**
     * Read bytes from the stream, consuming retained peek data first.
     * @param uint8Array - Uint8Array (or Buffer) to store data read from stream in
     * @param mayBeLess - Allow short reads at EOF; may still wait and combine source chunks
     * @returns Number of bytes read; only this prefix of the target contains result bytes.
     * Unlike tokenizer methods, a reader may return retained partial data at known EOF even without mayBeLess.
     */
    read(uint8Array: Uint8Array, mayBeLess?: boolean): Promise<number>;
    close(): Promise<void>;
    /**
     * Abort any active asynchronous operation are active, abort those before they may have completed.
     */
    abort(): Promise<void>;
}
export declare abstract class AbstractStreamReader implements IStreamReader {
    private closePromise?;
    private readonly onCloseCallback?;
    protected endOfStream: boolean;
    protected interrupted: boolean;
    constructor(options?: {
        onClose?: OnClose;
    });
    /**
     * Store peeked data
     * @type {Array}
     */
    protected peekQueue: Uint8Array[];
    peek(uint8Array: Uint8Array, mayBeLess?: boolean): Promise<number>;
    read(buffer: Uint8Array, mayBeLess?: boolean): Promise<number>;
    /**
     * Read chunk from stream
     * @param buffer - Target Uint8Array (or Buffer) to store data read from stream in
     * @returns Number of bytes read
     */
    protected readFromPeekBuffer(buffer: Uint8Array): number;
    readRemainderFromStream(buffer: Uint8Array, mayBeLess: boolean): Promise<number>;
    /**
     * Read from stream
     * @param buffer - Target Uint8Array (or Buffer) to store data read from stream in
     * @param mayBeLess - If true, may fill the buffer partially
     * @protected Bytes read
     */
    protected abstract readFromStream(buffer: Uint8Array, mayBeLess: boolean): Promise<number>;
    /** Share resource cleanup, callback completion and failures across close calls. */
    close(): Promise<void>;
    /** Backend cleanup completes before the optional close callback. */
    protected closeResource(): Promise<void>;
    /**
     * Abort any active asynchronous operation are active, abort those before they may have completed.
     */
    abstract abort(): Promise<void>;
}
