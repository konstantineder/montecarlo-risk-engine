import type { OnClose } from '../types.js';
import { AbstractStreamReader } from './AbstractStreamReader.js';
interface WebStreamByteReader {
    read(view: Uint8Array<ArrayBuffer>, options?: {
        min?: number;
    }): Promise<{
        done: boolean;
        value?: Uint8Array<ArrayBuffer>;
    }>;
    releaseLock(): void;
}
export declare abstract class WebStreamReader extends AbstractStreamReader {
    protected reader: WebStreamByteReader;
    constructor(reader: WebStreamByteReader, options?: {
        onClose?: OnClose;
    });
    /**
     * Read from stream
     * @param buffer - Target Uint8Array (or Buffer) to store data read from stream in
     * @param mayBeLess - If true, may fill the buffer partially
     * @protected Bytes read
     */
    protected abstract readFromStream(buffer: Uint8Array, mayBeLess: boolean): Promise<number>;
    private released;
    abort(): Promise<void>;
}
export {};
