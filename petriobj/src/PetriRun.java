import java.io.File;
import java.util.ArrayList;
import java.util.Locale;
import ua.stetsenkoinna.petriobj.ArcIn;
import ua.stetsenkoinna.petriobj.ArcOut;
import ua.stetsenkoinna.petriobj.PetriNet;
import ua.stetsenkoinna.petriobj.PetriObjModel;
import ua.stetsenkoinna.petriobj.PetriP;
import ua.stetsenkoinna.petriobj.PetriSim;
import ua.stetsenkoinna.petriobj.PetriT;
import ua.stetsenkoinna.pnml.PnmlParser;

/**
 * Прогін PNML-мережі рушієм PetriObjModel без GUI.
 * Аргументи: час моделювання, кількість прогонів, файли .pnml.
 * Вихід (JSON): для кожного файлу і прогону – середнє і кінцеве маркування позицій (p)
 * та середня і кінцева кількість зайнятих каналів переходів (t).
 */
public class PetriRun {
    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        double time = Double.parseDouble(args[0]);
        int reps = Integer.parseInt(args[1]);
        StringBuilder out = new StringBuilder("{");
        for (int f = 2; f < args.length; f++) {
            if (f > 2) out.append(',');
            out.append('"').append(args[f]).append("\":[");
            for (int r = 0; r < reps; r++) {
                // мережу читаємо заново: прогін змінює маркування і статистику;
                // парсер нумерує елементи статичними лічильниками, тож їх треба скинути
                PetriP.initNext();
                PetriT.initNext();
                ArcIn.initNext();
                ArcOut.initNext();
                PetriNet net = new PnmlParser().parse(new File(args[f]));
                ArrayList<PetriSim> list = new ArrayList<>();
                list.add(new PetriSim(net));
                PetriObjModel model = new PetriObjModel(list);
                model.setIsProtokol(false);
                model.go(time);
                if (r > 0) out.append(',');
                out.append("{\"p\":{");
                boolean first = true;
                for (PetriP p : net.getListP()) {
                    if (!first) out.append(',');
                    first = false;
                    out.append(String.format("\"%s\":[%.6f,%d]", p.getName(), p.getMean(), p.getMark()));
                }
                out.append("},\"t\":{");
                first = true;
                for (PetriT t : net.getListT()) {
                    if (!first) out.append(',');
                    first = false;
                    out.append(String.format("\"%s\":[%.6f,%d]", t.getName(), t.getMean(), t.getBuffer()));
                }
                out.append("}}");
            }
            out.append(']');
        }
        System.out.println(out.append('}'));
    }
}
