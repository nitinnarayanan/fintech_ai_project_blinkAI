from faker import Faker
from lxml import etree
import random, uuid, os

fake = Faker()

def generate_transaction_xml():
    """Create a synthetic ISO-20022-like transaction (pacs.008 style)."""
    # Root element
    root = etree.Element("Pacs008")

    # Unique ID
    etree.SubElement(root, "TxId").text = str(uuid.uuid4())

    # Payment info
    etree.SubElement(root, "Sender").text = fake.iban()
    etree.SubElement(root, "Receiver").text = fake.iban()
    etree.SubElement(root, "Amount").text = f"{round(random.uniform(10, 5000),2)}"
    etree.SubElement(root, "Currency").text = random.choice(["USD", "EUR", "GBP"])
    etree.SubElement(root, "Timestamp").text = fake.iso8601()
    etree.SubElement(root, "Country").text = fake.country_code()
    etree.SubElement(root, "Purpose").text = random.choice([
        "invoice payment","salary","refund","crypto purchase","loan","gift"
    ])
    etree.SubElement(root, "Channel").text = random.choice([
        "mobile","online","branch","atm"
    ])
    return etree.tostring(root, pretty_print=True).decode()

def save_transactions(n=100, output_dir="../data/transactions"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n):
        xml_str = generate_transaction_xml()
        with open(f"{output_dir}/tx_{i+1:03d}.xml", "w") as f:
            f.write(xml_str)

if __name__ == "__main__":
    save_transactions(200)
    print("✅ 200 synthetic ISO20022 transactions generated.")
