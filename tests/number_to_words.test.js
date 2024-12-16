import { describe, expect, test } from "vitest";
import { number_to_words } from "../outetts.js/version/v1/utils/number_to_words.js";

const TEST_CASES = {
    "Single digits": [
        [0, "zero"],
        [1, "one"],
        [2, "two"],
        [3, "three"],
        [4, "four"],
        [5, "five"],
        [6, "six"],
        [7, "seven"],
        [8, "eight"],
        [9, "nine"],
    ],
    "Two-digit numbers": [
        [10, "ten"],
        [11, "eleven"],
        [12, "twelve"],
        [13, "thirteen"],
        [15, "fifteen"],
        [19, "nineteen"],
        [20, "twenty"],
        [21, "twenty-one"],
        [42, "forty-two"],
        [81, "eighty-one"],
        [99, "ninety-nine"],
    ],
    "Triple-digit numbers": [
        [100, "one hundred"],
        [123, "one hundred and twenty-three"],
        [999, "nine hundred and ninety-nine"],
    ],
    "Large integers": [
        [1000, "one thousand"],
        [1234567890, "one billion, two hundred and thirty-four million, five hundred and sixty-seven thousand, eight hundred and ninety"],
        [9876543210, "nine billion, eight hundred and seventy-six million, five hundred and forty-three thousand, two hundred and ten"],
    ],
    Decimals: [
        [1, "one"],
        [1.0, "one"],
        [0.5, "zero point five"],
        [123.45, "one hundred and twenty-three point four five"],
        [3.1415926535, "three point one four one five nine two six five three five"],
        [0.1, "zero point one"],
        [0.01, "zero point zero one"],
        [0.001, "zero point zero zero one"],
        [0.0001, "zero point zero zero zero one"],
        [0.00001, "zero point zero zero zero zero one"],
        [0.000001, "zero point zero zero zero zero zero one"],
        [0.0000001, "zero point zero zero zero zero zero zero one"],
        [0.1000000023, "zero point one zero zero zero zero zero zero zero two three"],
        [0.987654, "zero point nine eight seven six five four"],
        [9.87654, "nine point eight seven six five four"],
        [98.7654, "ninety-eight point seven six five four"],
        [987.654, "nine hundred and eighty-seven point six five four"],
        [9876.54, "nine thousand, eight hundred and seventy-six point five four"],
        [98765.4, "ninety-eight thousand, seven hundred and sixty-five point four"],
    ],
    "Large decimal precision": [
        [3.141592653589793, "three point one four one five nine two six five three five eight nine seven nine three"],
        [2.718281828459045, "two point seven one eight two eight one eight two eight four five nine zero four five"],
        [1.414213562373095, "one point four one four two one three five six two three seven three zero nine five"],
    ],
    "Negative numbers": [
        [-1, "minus one"],
        [-123, "minus one hundred and twenty-three"],
        [-0.5, "minus zero point five"],
        [-999, "minus nine hundred and ninety-nine"],
    ],
    "Round numbers": [
        [10000, "ten thousand"],
        [1000000, "one million"],
        [1000000000, "one billion"],
        [1000000000000, "one trillion"],
    ],
    "Numbers just above round figures": [
        [1001, "one thousand and one"],
        [10001, "ten thousand and one"],
        [100001, "one hundred thousand and one"],
        [1000001, "one million and one"],
        [10000001, "ten million and one"],
        [100000001, "one hundred million and one"],
        [1000000001, "one billion and one"],
        [10000000001, "ten billion and one"],
        [100000000001, "one hundred billion and one"],
        [1000000000001, "one trillion and one"],
    ],
    "Mixed digits": [
        [105, "one hundred and five"],
        [1001, "one thousand and one"],
        [1001000000001, "one trillion, one billion and one"],
        [1001000000100, "one trillion, one billion, one hundred"],
        [101010, "one hundred and one thousand and ten"],
        [123456, "one hundred and twenty-three thousand, four hundred and fifty-six"],
        [555123, "five hundred and fifty-five thousand, one hundred and twenty-three"],
    ],
    "Edge cases": [
        [-0, "minus zero"],
        [999999999999, "nine hundred and ninety-nine billion, nine hundred and ninety-nine million, nine hundred and ninety-nine thousand, nine hundred and ninety-nine"],
        [Number.MAX_SAFE_INTEGER, "nine quadrillion, seven trillion, one hundred and ninety-nine billion, two hundred and fifty-four million, seven hundred and forty thousand, nine hundred and ninety-one"],
    ],
};
const UNSAFE_NUMBERS = [Number.MAX_SAFE_INTEGER + 1, Number.MIN_SAFE_INTEGER - 1, Number.MAX_VALUE, NaN, Infinity, -Infinity, null, undefined, ""];

describe("Number to Words Tests", () => {
    Object.keys(TEST_CASES).forEach((testName) => {
        test(testName, () => {
            TEST_CASES[testName].forEach(([input, expected]) => {
                expect(number_to_words(input)).toEqual(expected);
            });
        });
    });

    test("Unsafe numbers", () => {
        UNSAFE_NUMBERS.forEach((num) => {
            expect(() => number_to_words(num)).toThrow();
        });
    });
});
