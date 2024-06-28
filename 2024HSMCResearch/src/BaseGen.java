
import java.util.ArrayList;

public class BaseGen {
    //Add connectedness
    public static void print(ArrayList<ArrayList<Integer>> graph){
        for (int i=0;i<graph.size();i++){
            System.out.println(i+": "+graph.get(i));
        }
    }

    public static boolean checkTriangle(ArrayList<ArrayList<Integer>> graph){
        int vertices = graph.size();
        for (int i=0;i<vertices;i++){
            for (int j=i+1;j<vertices;j++){
                for (int k=0;k<vertices;k++){
                    if (k==i || k==j){
                        continue;
                    }
                    if(graph.get(i).contains(k) && graph.get(j).contains(k) && graph.get(i).contains(j)){
                        return true;
                    }
                }
            }
        }
        return false;
    }

    public static boolean checkMinimal(ArrayList<ArrayList<Integer>> graph, int a, int b, int c){
        int vertices = graph.size();
        for (int i=0; i<vertices;i++){
            for (int j=i+1;j<vertices;j++){
                if (graph.get(i).contains(j)){
                    continue;
                }
                if (i<a && j<a){
                    continue;
                }
                else if (i<a+b && i>=a && j<a+b && j>=a){
                    continue;
                }
                else if (i>=a+b && j>=a+b){
                    continue;
                }
                ArrayList<ArrayList<Integer>> current = new ArrayList<>();
                for (int k=0;k<vertices;k++){
                    
                    current.add((ArrayList)graph.get(k).clone());
                }
                
                current.get(i).add(j);
                current.get(j).add(i);
                if (!checkTriangle(current)){
                    return false;
                }
            }
        }
        return true;
    }

    public static void main(String[] args) throws Exception {
        //System.out.println("test");
        ArrayList<ArrayList<Integer>> check =new ArrayList<>();
        ArrayList <Integer> c0 = new ArrayList<>();
        c0.add(1);
        //c0.add(2);
        //c0.add(3);
        c0.add(4);

        ArrayList <Integer> c1 = new ArrayList<>();
        c1.add(0);
        c1.add(2);
        //c1.add(3);
        //c1.add(4);

        ArrayList <Integer> c2 = new ArrayList<>();
        c2.add(1);
        c2.add(3);

        ArrayList <Integer> c3 = new ArrayList<>();
        c3.add(2);
        c3.add(4);

        ArrayList <Integer> c4 = new ArrayList<>();
        c4.add(3);
        c4.add(0);

        check.add(c0);
        check.add(c1);
        check.add(c2);
        check.add(c3);
        check.add(c4);
        System.out.println(checkTriangle(check));
        System.out.println(checkMinimal(check, 1, 2, 2));
//        print(check);


        int maxVertices = 5;


        //considering rotating colors to be the same
        //generating graph
        for (int i=5;i<=maxVertices;i++){
            for (int a=1; a<=i;a++){ //can start at a=0
                for (int b=a;a+b<=i;b++){
                    ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
                    for (int j=0;j<i;j++){
                        graph.add(new ArrayList<>());
                    }

                    int c = i-b-a;
                    if (c<a || c<b){
                        continue;
                    }
                    if (a==1 && b==2 &&c==2){
                        System.out.println("here");
                    }
                    else{
                        continue;
                    }
                    int ab = a*b; //number of potential edges between colors a and b
                    int ac = a*c;
                    int bc = b*c;
                    int vIndex=0;
                    for (int j=0;j<Math.pow(2,ab);j++){
                        //System.out.println(graph.get(0).size());
                        // System.out.println("here");
                        String currentab = Integer.toBinaryString(j);
                        String add = "";
                        for (int prepend=0;prepend<ab-currentab.length();prepend++){
                            add+="0";
                            System.out.println("here2");

                        }
                        currentab = add+currentab;
                        if (currentab.length()>ab){
                            System.out.println("wtf");
                        }
                        System.out.println(a+" "+b+" " +c+" "+currentab);
                        //int countAB=0;
                        for (int k=currentab.length()-1;k>=0;k--){
                            if (currentab.charAt(k)=='1'){
                                graph.get((currentab.length()-1-k)/b).add(a+(currentab.length()-1-k)%b);
                                graph.get(a+(currentab.length()-1-k)%b).add((currentab.length()-1-k)/b);
                            }
                        }

                        for (int l=0;l<Math.pow(2,ac);l++){
                            String currentac = Integer.toBinaryString(l);
                            add = "";
                            for (int prepend=0;prepend<ac-currentac.length();prepend++){
                            add+="0";
                            //System.out.println("here");

                            }
                            currentac = add+currentac;
                            if (currentac.length()>ac){
                            System.out.println("wtf");
                            }
                            System.out.println("ac: "+currentac);
                            //int countB=0;
                            for (int m=currentac.length()-1;m>=0;m--){
                                if (currentac.charAt(m)=='1'){
                                    graph.get((currentac.length()-1-m)/c).add(a+b+(currentac.length()-1-m)%c);
                                    graph.get(a+b+(currentac.length()-1-m)%c).add((currentac.length()-1-m)/c);
                                }
                            }
                            
                            for (int n=0;n<Math.pow(2,bc);n++){
                                String currentbc = Integer.toBinaryString(n);
                                //int countB=0;
                                add = "";
                                for (int prepend=0;prepend<bc-currentbc.length();prepend++){
                                add+="0";
                                //System.out.println("here2");

                                }
                                currentbc = add+currentbc;
                                if (currentbc.length()>bc){
                                    System.out.println("wtf3");
                                }
                                System.out.println("bc: " + currentbc);
                                for (int p=currentbc.length()-1;p>=0;p--){
                                    if (currentbc.charAt(p)=='1'){
                                        graph.get(a+(currentbc.length()-1-p)/c).add(a+b+(currentbc.length()-1-p)%c);
                                        graph.get(a+b+(currentbc.length()-1-p)%c).add(a+(currentbc.length()-1-p)/c);
                                    }
                                }
                                //print(graph);
                                //System.out.println();
                                if(check.equals(graph)){
                                    System.out.println("yay");
                                }
                                if (!checkTriangle(graph) && checkMinimal(graph, a, b, c)){
                                    System.out.println(a +" "+b+" "+c);
                                    print(graph);
                                    System.out.println("did");
                                }
                                graph.clear();
                                for (int cl=0;cl<i;cl++){
                                    graph.add(new ArrayList<>());
                                }
                                
                                
                            }
                        }
                    }

                }
            }
        }

    }

}
