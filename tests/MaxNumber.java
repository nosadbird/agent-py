import java.util.*;

public class MaxNumber {
    public static void main(String[] args) {
        int n = 213;
        int[] a = {1, 2, 3};
        System.out.println(findMaxNumber(n, a)); // Expected: 213

        n = 210;
        a = new int[]{1, 2, 3};
        System.out.println(findMaxNumber(n, a)); // Expected: 133

        n = 100;
        a = new int[]{1, 0};
        System.out.println(findMaxNumber(n, a)); // Expected: 100
    }

    public static int findMaxNumber(int n, int[] a) {
        Set<Integer> set = new HashSet<>();
        for (int num : a) set.add(num);
        List<Integer> digits = new ArrayList<>(set);
        Collections.sort(digits, Collections.reverseOrder());

        String s = String.valueOf(n);
        int len = s.length();

        StringBuilder sb = new StringBuilder();
        if (construct(sb, s, digits, 0)) {
            return Integer.parseInt(sb.toString());
        }

        if (len > 1) {
            char maxDigit = (char)('0' + digits.get(0));
            return Integer.parseInt(String.valueOf(maxDigit).repeat(len - 1));
        }
        return -1;
    }

    private static boolean construct(StringBuilder sb, String s, List<Integer> digits, int index) {
        if (index == s.length()) return true;

        int currentDigit = s.charAt(index) - '0';
        for (int d : digits) {
            if (d > currentDigit) continue;
            
            if (d < currentDigit) {
                sb.append(d);
                for (int i = index + 1; i < s.length(); i++) {
                    sb.append(digits.get(0));
                }
                return true;
            }

            sb.append(d);
            if (construct(sb, s, digits, index + 1)) {
                return true;
            }
            sb.setLength(sb.length() - 1);
        }
        return false;
    }
}