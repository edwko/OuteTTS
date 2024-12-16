import en_female_1 from "./default_speakers/en_female_1.json" with { type: "json" };
import en_female_2 from "./default_speakers/en_female_2.json" with { type: "json" };
import en_male_1 from "./default_speakers/en_male_1.json" with { type: "json" };
import en_male_2 from "./default_speakers/en_male_2.json" with { type: "json" };
import en_male_3 from "./default_speakers/en_male_3.json" with { type: "json" };
import en_male_4 from "./default_speakers/en_male_4.json" with { type: "json" };
import ja_female_1 from "./default_speakers/ja_female_1.json" with { type: "json" };
import ja_female_2 from "./default_speakers/ja_female_2.json" with { type: "json" };
import ja_female_3 from "./default_speakers/ja_female_3.json" with { type: "json" };
import ja_male_1 from "./default_speakers/ja_male_1.json" with { type: "json" };
import ko_female_1 from "./default_speakers/ko_female_1.json" with { type: "json" };
import ko_female_2 from "./default_speakers/ko_female_2.json" with { type: "json" };
import ko_male_1 from "./default_speakers/ko_male_1.json" with { type: "json" };
import ko_male_2 from "./default_speakers/ko_male_2.json" with { type: "json" };
import zh_female_1 from "./default_speakers/zh_female_1.json" with { type: "json" };
import zh_male_1 from "./default_speakers/zh_male_1.json" with { type: "json" };

export default {
    en: {
        male_1: en_male_1,
        male_2: en_male_2,
        male_3: en_male_3,
        male_4: en_male_4,
        female_1: en_female_1,
        female_2: en_female_2,
    },
    ja: {
        male_1: ja_male_1,
        female_1: ja_female_1,
        female_2: ja_female_2,
        female_3: ja_female_3,
    },
    ko: {
        male_1: ko_male_1,
        male_2: ko_male_2,
        female_1: ko_female_1,
        female_2: ko_female_2,
    },
    zh: {
        male_1: zh_male_1,
        female_1: zh_female_1,
    },
};
