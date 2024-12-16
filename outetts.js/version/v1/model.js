export class HFModel {
    constructor(model) {
        this.model = model;
    }

    async generate(...args) {
        return (await this.model.generate(...args)).tolist()[0];
    }
}
