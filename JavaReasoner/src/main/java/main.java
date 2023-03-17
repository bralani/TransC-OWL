import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;

public class main {

    public static void main (String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        Map<String, Integer> dictionary = new HashMap<String, Integer>();
        dictionary.put("equivalentClass", 0);
        dictionary.put("equivalentProperty", 1);
        dictionary.put("functionalProperty", 2);
        dictionary.put("inverseOf", 3);
        dictionary.put("subClassOf", 4);
        dictionary.put("rangeDomain", 5);
        dictionary.put("DisjointWith", 6);
        dictionary.put("classList", 7);
        dictionary.put("merge", 8);
        dictionary.put("instanceOf", 9);

        reasoner r = new reasoner("Nellontology.ttl","triples_nell.ttl","TURTLE");

        switch(7) {
            case 0:
                r.generateEquivalentClass();
                break;
            case 1:
                r.generateEquivalentPropertyClass();
                break;
            case 2:
                r.generateFunctionalPropertyClass();
                break;
            case 3:
                r.generateInverseOfClass();
                break;
            case 4:
                r.generateSubClassOf();
                break;
            case 5:
                r.rangeDomain();
                r.getRangeDomainFromSer();
                break;
            case 6:
                r.generateDisjointClass();
                r.corruptClass();
                break;
            case 7:
                r.class2id();
                break;
            case 8:
                r.yagoMerge(args[4],args[5],args[6],args[7],args[8]);
                break;
            case 9:
                r.generateInstanceOf();
                break;
        }
    }
}
