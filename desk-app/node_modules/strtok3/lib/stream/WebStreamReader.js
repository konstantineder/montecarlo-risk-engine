import { AbstractStreamReader } from './AbstractStreamReader.js';
export class WebStreamReader extends AbstractStreamReader {
    constructor(reader, options) {
        super(options);
        this.reader = reader;
        this.released = false;
    }
    async abort() {
        if (!this.released) {
            this.reader.releaseLock();
            this.released = true;
        }
    }
}
