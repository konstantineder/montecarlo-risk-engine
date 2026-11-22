import type { ReadableStream as NodeReadableStream } from 'node:stream/web';
import type { OnClose } from '../types.js';
import { WebStreamByobReader } from './WebStreamByobReader.js';
import { WebStreamDefaultReader } from './WebStreamDefaultReader.js';
export type AnyWebByteStream = NodeReadableStream<Uint8Array> | ReadableStream<Uint8Array>;
export declare function makeWebStreamReader(stream: AnyWebByteStream, options?: {
    onClose?: OnClose;
}): WebStreamByobReader | WebStreamDefaultReader;
